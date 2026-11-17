use crate::{FlyCameraController, RenderParams};
use std::time::{Duration, Instant};

pub struct Gui {
    pub context: egui::Context,
    renderer: egui_wgpu::Renderer,
    frame: Option<UiFrame>,
    next_repaint: Instant,
    input_geometry: (Option<egui::Rect>, Option<f32>, bool),
}

struct UiFrame {
    paint_jobs: Vec<egui::ClippedPrimitive>,
    screen: egui_wgpu::ScreenDescriptor,
    free_textures: Vec<egui::TextureId>,
}

impl Gui {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        Self {
            context: egui::Context::default(),
            renderer: egui_wgpu::Renderer::new(device, format, Default::default()),
            frame: None,
            next_repaint: Instant::now(),
            input_geometry: (None, None, false),
        }
    }

    pub fn run(
        &mut self,
        input: egui::RawInput,
        params: &mut RenderParams,
        camera: &mut FlyCameraController,
        fps: f32,
        progress: f32,
    ) -> Option<egui::FullOutput> {
        let now = Instant::now();
        let geometry = (
            input.screen_rect,
            input
                .viewports
                .get(&egui::ViewportId::ROOT)
                .and_then(|v| v.native_pixels_per_point),
            input.focused,
        );
        if self.frame.is_some()
            && input.events.is_empty()
            && geometry == self.input_geometry
            && now < self.next_repaint
            && !self.context.has_requested_repaint()
        {
            return None;
        }
        self.input_geometry = geometry;
        let output = self.context.run_ui(input, |ui| {
            draw_controls(ui.ctx(), params, camera, fps, progress);
        });
        // Scene rendering continues independently. Status labels update at 10 Hz;
        // pointer/keyboard events and egui animation deadlines bypass this limit.
        let delay = output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map_or(Duration::ZERO, |viewport| viewport.repaint_delay);
        self.next_repaint = now + delay.min(Duration::from_millis(100));
        Some(output)
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        output: Option<egui::FullOutput>,
        size: [u32; 2],
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(mut output) = output else {
            return Vec::new();
        };
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: size,
            pixels_per_point: output.pixels_per_point,
        };
        for (id, deltas) in output.textures_delta.set.drain() {
            for delta in deltas {
                self.renderer.update_texture(device, queue, id, &delta);
            }
        }
        let paint_jobs = self
            .context
            .tessellate(output.shapes, output.pixels_per_point);
        let command_buffers =
            self.renderer
                .update_buffers(device, queue, encoder, &paint_jobs, &screen);
        self.frame = Some(UiFrame {
            paint_jobs,
            screen,
            free_textures: output.textures_delta.free.drain().collect(),
        });
        command_buffers
    }

    pub fn render(&self, pass: &mut wgpu::RenderPass<'static>) {
        let frame = self.frame.as_ref().expect("prepare UI before rendering");
        self.renderer.render(pass, &frame.paint_jobs, &frame.screen);
    }

    pub fn free_textures(&mut self) {
        if let Some(frame) = &mut self.frame {
            for id in frame.free_textures.drain(..) {
                self.renderer.free_texture(&id);
            }
        }
    }
}

fn draw_controls(
    context: &egui::Context,
    params: &mut RenderParams,
    camera: &mut FlyCameraController,
    fps: f32,
    progress: f32,
) {
    egui::Window::new("Parameters")
        .default_pos([60., 60.])
        .default_size([300., 300.])
        .vscroll(true)
        .show(context, |ui| {
            ui.label(format!(
                "FPS: {fps:.1}, render progress: {:.1} %",
                progress * 100.
            ));
            ui.separator();
            ui.label("Sampling parameters");
            ui.label("samples per pixel per frame");
            ui.horizontal(|ui| {
                for samples in [1, 8, 16] {
                    ui.radio_value(
                        &mut params.sampling.num_samples_per_pixel,
                        samples,
                        samples.to_string(),
                    );
                }
            });
            ui.label("total samples per pixel");
            ui.horizontal(|ui| {
                for samples in [128, 256, 512] {
                    ui.radio_value(
                        &mut params.sampling.max_samples_per_pixel,
                        samples,
                        samples.to_string(),
                    );
                }
            });
            ui.add(egui::Slider::new(&mut params.sampling.num_bounces, 1..=10).text("num bounces"));
            ui.separator();
            ui.label("Denoiser Parameters");
            ui.checkbox(&mut params.sampling.denoising_enabled, "Enable Denoiser");
            if params.sampling.denoising_enabled {
                ui.add(
                    egui::Slider::new(&mut params.sampling.denoising_radius, 0..=5)
                        .text("Denoising Radius"),
                );
                ui.add(
                    egui::Slider::new(&mut params.sampling.denoising_sigma_color, 0.01..=1.)
                        .text("Denoising Sigma Color"),
                );
            }
            ui.separator();
            ui.label("Camera parameters");
            ui.checkbox(&mut params.auto_exposure, "auto exposure");
            ui.add(
                egui::Slider::new(&mut params.exposure, 0.01..=32.)
                    .logarithmic(true)
                    .text(if params.auto_exposure {
                        "exposure compensation"
                    } else {
                        "exposure"
                    }),
            );
            ui.add(egui::Slider::new(&mut camera.vfov_degrees, 10.0..=90.0).text("vfov"));
            ui.add(egui::Slider::new(&mut camera.aperture, 0.0..=1.0).text("aperture"));
            ui.add(
                egui::Slider::new(&mut camera.focus_distance, 5.0..=20.0).text("focus distance"),
            );
            ui.separator();
            ui.label("Sky parameters");
            ui.add(egui::Slider::new(&mut params.sky.azimuth_degrees, 0.0..=360.0).text("azimuth"));
            ui.add(
                egui::Slider::new(&mut params.sky.zenith_degrees, 0.0..=90.0).text("inclination"),
            );
            ui.add(egui::Slider::new(&mut params.sky.turbidity, 1.0..=10.0).text("turbidity"));
            ui.horizontal(|ui| {
                ui.label("albedo");
                ui.color_edit_button_rgb(&mut params.sky.albedo);
            });
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_controls_have_real_font_glyphs_without_system_fonts() {
        let context = egui::Context::default();
        context.set_missing_glyph_policy(egui::MissingGlyphPolicy::Panic);
        let mut camera = FlyCameraController::default();
        let mut params = RenderParams {
            camera: camera.renderer_camera(),
            sampling: crate::SamplingParams::default(),
            exposure: 0.246,
            auto_exposure: true,
            sky: crate::SkyParams::default(),
            viewport_size: (640, 480),
        };
        let output = context.run_ui(egui::RawInput::default(), |ui| {
            draw_controls(ui.ctx(), &mut params, &mut camera, 60., 0.5);
            ui.ctx().fonts_mut(|fonts| {
                let font = egui::FontId::proportional(14.);
                assert!(fonts.has_glyphs(&font, "Parameters FPS 0123456789"));
                let narrow = fonts.layout_no_wrap("I".into(), font.clone(), egui::Color32::WHITE);
                let wide = fonts.layout_no_wrap("W".into(), font, egui::Color32::WHITE);
                assert!(
                    narrow.size().x < wide.size().x,
                    "text must not be equal-width missing-glyph boxes"
                );
            });
        });
        output.drop_without_applying_deltas();
    }
}
