//! Training-data capture for the temporal neural denoiser.
//!
//!     weekend-raytracer-wgpu --capture DIR --model SCENE.glb --camera x,y,z,dx,dy,dz,fov
//!         [--scale S] [--sequences N] [--frames F] [--reference-spp R] [--size WxH] [--seed K] [--motion-scale M | --motion-range LO,HI]
//!
//! Each sequence moves the camera (translation, yaw and pitch velocities; some
//! sequences stay still) and four spheres with fixed random materials. For every
//! frame it writes `DIR/seq_NNN/frame_MM.f16`: per pixel, little-endian f16
//! [noisy rgb (1 spp), guides (12, see raytracer.wgsl `gbuffer`), reference rgb].
//! `scale` is the scene's typical feature size in scene units (about 1 for
//! metre-scale interiors); it sets camera speed and sphere sizes.
//! `--crop N` traces and stores only an N x N window per sequence (random
//! position, recorded in seq_NNN/crop.json), so high-resolution, high-spp
//! references cost a fraction of a full frame.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
use crate::{Material, Raytracer, RenderParams, SamplingParams, Sphere, Texture};
use nalgebra_glm as glm;

/// xorshift64*: deterministic per seed, no extra dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32) / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
    fn unit(&mut self) -> glm::Vec3 {
        loop {
            let v = glm::vec3(
                self.range(-1., 1.),
                self.range(-1., 1.),
                self.range(-1., 1.),
            );
            if glm::length2(&v) <= 1. && glm::length2(&v) > 1e-4 {
                return glm::normalize(&v);
            }
        }
    }
}

const SPHERES: usize = 4;

pub fn run() {
    pollster::block_on(capture());
}

fn number<T: std::str::FromStr>(name: &str, default: T) -> T {
    crate::argument_value(name).map_or(default, |v| {
        v.parse()
            .unwrap_or_else(|_| panic!("{name} expects a number"))
    })
}

async fn capture() {
    let output = std::path::PathBuf::from(crate::argument_value("--capture").unwrap());
    let scale: f32 = number("--scale", 1.);
    let sequences: u32 = number("--sequences", 8);
    let frames: u32 = number("--frames", 8);
    // Per-frame motion multiplier (1 = training motion; ~0.1 for smooth 60 fps video).
    let motion_scale: f32 = number("--motion-scale", 1.);
    // `--motion-range LO,HI`: per-sequence motion multiplier, log-uniform in [LO, HI].
    let motion_range: Option<(f32, f32)> = crate::argument_value("--motion-range").map(|v| {
        let (lo, hi) = v.split_once(',').expect("--motion-range LO,HI");
        (lo.parse().unwrap(), hi.parse().unwrap())
    });
    let digits = frames.saturating_sub(1).to_string().len().max(2);
    let reference_spp: u32 = number("--reference-spp", 128);
    let seed: u64 = number("--seed", 1);
    let size = crate::argument_value("--size").unwrap_or_else(|| "320x200".into());
    let (w, h) = size.split_once('x').expect("--size WxH");
    let (width, height): (u32, u32) = (w.parse().unwrap(), h.parse().unwrap());
    let crop: Option<u32> = crate::argument_value("--crop").map(|v| v.parse().expect("--crop N"));
    let (base_eye, base_dir, base_fov) =
        crate::camera_argument().expect("--capture needs --camera (the scene's base view)");

    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut scene = crate::selected_scene();
    // Fixed per run: the material list and sphere count are pipeline constants.
    for i in 0..SPHERES {
        let color = glm::vec3(
            rng.range(0.1, 0.95),
            rng.range(0.1, 0.95),
            rng.range(0.1, 0.95),
        );
        scene.materials.push(match (i + seed as usize) % 4 {
            0 => Material::Lambertian {
                albedo: Texture::new_from_color(color),
            },
            1 => Material::Metal {
                albedo: Texture::new_from_color(color),
                fuzz: rng.range(0., 0.4),
            },
            2 => Material::Dielectric {
                refraction_index: 1.5,
            },
            _ => Material::Checkerboard {
                even: Texture::new_from_color(color),
                odd: Texture::new_from_color(glm::vec3(0.9, 0.9, 0.9) - color * 0.8),
            },
        });
        scene
            .spheres
            .push(Sphere::new(base_eye, scale * 0.3, i as u32));
    }

    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (features, limits) = crate::raytracer::gpu_textures::device_requirements(&adapter);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: features | crate::experiment_features(&adapter),
            required_limits: limits,
            experimental_features: crate::raytracer::gpu_textures::experimental_features(),
            ..Default::default()
        })
        .await
        .unwrap();
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        width,
        height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    };
    let mut params = RenderParams {
        camera: crate::scene_camera(&scene).renderer_camera(),
        exposure: 2.,
        auto_exposure: false,
        sky: crate::scene_sky(&scene),
        sampling: SamplingParams {
            max_samples_per_pixel: reference_spp,
            num_bounces: crate::selected_bounces(),
            denoising_enabled: false,
            ..Default::default()
        },
        viewport_size: (width, height),
    };
    params.camera.aperture = 0.;
    let mut tracer = Raytracer::new(&device, &queue, &config, &scene, &params).unwrap();
    tracer.set_skip_filter(!crate::has_flag("--temporal-model"));
    let (out_w, out_h) = crop.map_or((width, height), |n| (n.min(width), n.min(height)));
    let start = std::time::Instant::now();

    let up = glm::vec3(0., 1., 0.);
    for sequence in 0..sequences {
        let directory = output.join(format!("seq_{sequence:03}"));
        std::fs::create_dir_all(&directory).unwrap();
        let origin = if crop.is_some() {
            let x = ((rng.next() * (width - out_w + 1) as f32) as u32).min(width - out_w);
            let y = ((rng.next() * (height - out_h + 1) as f32) as u32).min(height - out_h);
            tracer.set_trace_region(Some((x, y, out_w, out_h)));
            (x, y)
        } else {
            (0, 0)
        };
        std::fs::write(
            directory.join("crop.json"),
            format!(
                "{{\"x\":{},\"y\":{},\"frame_width\":{width},\"frame_height\":{height}}}",
                origin.0, origin.1
            ),
        )
        .unwrap();
        // Camera start: near the base view, turned a little.
        let base_yaw = base_dir.z.atan2(base_dir.x);
        let base_pitch = base_dir.y.asin();
        let right = glm::normalize(&glm::cross(&base_dir, &up));
        let mut eye = base_eye
            + right * scale * rng.range(-1.5, 1.5)
            + up * scale * rng.range(-0.3, 0.5)
            + base_dir * scale * rng.range(-1., 1.5);
        let mut yaw = base_yaw + rng.range(-0.45, 0.45);
        let mut pitch = (base_pitch + rng.range(-0.15, 0.15)).clamp(-1.2, 1.2);
        let fov = (base_fov + rng.range(-10., 10.)).clamp(35., 85.);
        // Motion: 15% still, otherwise translation and/or rotation.
        let still = rng.next() < 0.15;
        let motion = match motion_range {
            Some((lo, hi)) => (lo.ln() + rng.next() * (hi.ln() - lo.ln())).exp(),
            None => motion_scale,
        };
        let speed = if still {
            0.
        } else {
            motion * scale * rng.range(0., 0.2)
        };
        let velocity = {
            let mut d = rng.unit();
            d.y *= 0.3;
            glm::normalize(&d) * speed
        };
        let (yaw_rate, pitch_rate) = if still {
            (0., 0.)
        } else {
            (
                motion * rng.range(-0.05, 0.05),
                motion * rng.range(-0.015, 0.015),
            )
        };
        let forward = |yaw: f32, pitch: f32| {
            glm::vec3(
                yaw.cos() * pitch.cos(),
                pitch.sin(),
                yaw.sin() * pitch.cos(),
            )
        };
        let mut spheres = scene.spheres.clone();
        let mut sphere_velocity = Vec::new();
        let start_dir = forward(yaw, pitch);
        let start_right = glm::normalize(&glm::cross(&start_dir, &up));
        for sphere in &mut spheres {
            let radius = scale * rng.range(0.12, 0.45);
            let center = eye
                + start_dir * scale * rng.range(1.5, 5.)
                + start_right * scale * rng.range(-2., 2.)
                + up * scale * rng.range(-0.6, 0.8);
            *sphere = Sphere::new(center, radius, sphere_material(sphere));
            sphere_velocity.push(rng.unit() * motion * scale * rng.range(0., 0.15));
        }
        for frame in 0..frames {
            params.camera.eye_pos = eye;
            params.camera.eye_dir = forward(yaw, pitch);
            params.camera.vfov = crate::raytracer::Angle::degrees(fov);
            tracer.set_render_params(&device, &queue, &params).unwrap();
            tracer.set_spheres(&queue, &spheres);
            // Noisy 1-spp frame and its guides.
            params.sampling.num_samples_per_pixel = 1;
            tracer.set_render_params(&device, &queue, &params).unwrap();
            render(&device, &queue, &mut tracer, 1);
            let noisy = tracer.read_radiance(&device, &queue);
            let guides = tracer.read_gbuffer(&device, &queue);
            if crate::has_flag("--temporal-model") {
                // Parity checks: the renderer's denoised frame, f32 rgb.
                let denoised = tracer.read_filtered(&device, &queue);
                std::fs::write(
                    directory.join(format!("frame_{frame:0digits$}.den.f32")),
                    bytemuck::cast_slice(&denoised),
                )
                .unwrap();
            }
            // Reference: same instant, same camera, many samples.
            let reference = if crate::has_flag("--no-reference") {
                vec![0.; noisy.len()]
            } else {
                // Many samples per dispatch: one dispatch per sample would wait for
                // its slowest ray thousands of times per frame.
                let per_dispatch = [32, 16, 8, 4, 2, 1]
                    .into_iter()
                    .find(|k| reference_spp % k == 0)
                    .unwrap();
                params.sampling.num_samples_per_pixel = per_dispatch;
                tracer.set_render_params(&device, &queue, &params).unwrap();
                tracer.reset_accumulation();
                render(&device, &queue, &mut tracer, reference_spp / per_dispatch);
                tracer.read_radiance(&device, &queue)
            };
            let pixels = (out_w * out_h) as usize;
            let mut bytes = Vec::with_capacity(pixels * 18 * 2);
            for q in 0..pixels {
                let p =
                    ((origin.1 + q as u32 / out_w) * width + origin.0 + q as u32 % out_w) as usize;
                for v in noisy[p * 3..p * 3 + 3]
                    .iter()
                    .chain(&guides[p * 12..p * 12 + 12])
                    .chain(&reference[p * 3..p * 3 + 3])
                {
                    let v = if v.is_finite() {
                        v.clamp(-65000., 65000.)
                    } else {
                        0.
                    };
                    bytes.extend(half::f16::from_f32(v).to_le_bytes());
                }
            }
            std::fs::write(directory.join(format!("frame_{frame:0digits$}.f16")), bytes).unwrap();
            // Advance the camera and spheres for the next frame.
            eye += velocity;
            yaw += yaw_rate;
            pitch = (pitch + pitch_rate).clamp(-1.2, 1.2);
            for (sphere, v) in spheres.iter_mut().zip(&sphere_velocity) {
                sphere.set_center(sphere.center() + v);
            }
        }
        eprintln!(
            "capture: sequence {}/{sequences} done, {:.0}s elapsed",
            sequence + 1,
            start.elapsed().as_secs_f64()
        );
    }
    std::fs::write(
        output.join("meta.json"),
        format!(
            "{{\"width\":{out_w},\"height\":{out_h},\"frame_width\":{width},\"frame_height\":{height},\"frames\":{frames},\"sequences\":{sequences},\"reference_spp\":{reference_spp},\"scale\":{scale},\"channels\":[\"noisy_r\",\"noisy_g\",\"noisy_b\",\"albedo_r\",\"albedo_g\",\"albedo_b\",\"distance\",\"normal_x\",\"normal_y\",\"normal_z\",\"roughness\",\"motion_x\",\"motion_y\",\"metallic\",\"kind\",\"ref_r\",\"ref_g\",\"ref_b\"]}}"
        ),
    )
    .unwrap();
}

fn sphere_material(sphere: &Sphere) -> u32 {
    sphere.material_index()
}

/// Accumulate `samples` frames (one sample each) from a cleared accumulation.
fn render(device: &wgpu::Device, queue: &wgpu::Queue, tracer: &mut Raytracer, samples: u32) {
    for _ in 0..samples {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        tracer.accumulate_frame(queue, &mut encoder);
        queue.submit([encoder.finish()]);
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}
