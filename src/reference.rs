//! Reference frames for training and evaluation (`--reference-output DIR`), as close to
//! the converged image as a time budget allows. See docs/reference-frames.md.
//!
//!     weekend-raytracer-wgpu --reference-output DIR --model SCENE.glb --camera ... --size WxH
//!         [--reference-seconds T] [--reference-error E] [--reference-uniform N]
//!
//! Samples go into GROUPS independent groups, so each pixel has a mean and a measured
//! standard error. After a first pass over the frame, only the tiles whose error is
//! highest get more samples, until every tile is below `--reference-error` (standard
//! error of a pixel's luminance in exposure-normalized log space, 95th percentile of
//! the tile; default 0.002) or `--reference-seconds` (default 300) run out.
//! `--reference-uniform N`: the old method instead, N samples at every pixel.
//! Writes DIR/groups.f32 (per group: mean rgb per pixel, f32), DIR/spp.f32 (samples per
//! pixel), DIR/guides.f32 (pixel-integrated albedo and normal) and DIR/meta.json; training/reference/finish.py makes the final image.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
use crate::{Raytracer, RenderParams, SamplingParams};
use std::time::Instant;

const GROUPS: usize = 8;
/// Samples per pixel per dispatch. Small: a dispatch that runs for more than about half
/// a second is reset by the Intel driver (the work is silently dropped). Fast GPUs take
/// larger tiles instead (`--reference-tile`).
const PASS: u32 = 8;
/// Least dispatches per tile and group in a round (fewer, larger readbacks).
const ROUND: u32 = 4;

pub fn run() {
    pollster::block_on(render());
}

fn number<T: std::str::FromStr>(name: &str, default: T) -> T {
    crate::argument_value(name).map_or(default, |v| {
        v.parse()
            .unwrap_or_else(|_| panic!("{name} expects a number"))
    })
}

/// A tile: x, y, width, height.
type Tile = (u32, u32, u32, u32);

/// `passes` dispatches of PASS samples over a tile; its pixels then hold their sum.
fn trace(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tracer: &mut Raytracer,
    tile: Tile,
    passes: u32,
) {
    tracer.set_trace_region(Some(tile));
    tracer.reset_accumulation();
    for _ in 0..passes {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        tracer.accumulate_frame(queue, &mut encoder);
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
}

fn luminance(c: &[f32]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

async fn render() {
    let output = std::path::PathBuf::from(crate::argument_value("--reference-output").unwrap());
    std::fs::create_dir_all(&output).unwrap();
    let size = crate::argument_value("--size").unwrap_or_else(|| "1920x1080".into());
    let (w, h) = size.split_once('x').expect("--size WxH");
    let (width, height): (u32, u32) = (w.parse().unwrap(), h.parse().unwrap());
    let seconds: f64 = number("--reference-seconds", 300.);
    let target: f32 = number("--reference-error", 0.002);
    let uniform: Option<u32> = crate::argument_value("--reference-uniform").map(|v| {
        let n: u32 = v.parse().expect("--reference-uniform N");
        assert!(
            n >= PASS && n % PASS == 0,
            "--reference-uniform: a multiple of {PASS}"
        );
        n
    });
    let tile_size: u32 = number("--reference-tile", 120);
    // `--reference-window X,Y,W,H`: only the tiles starting inside this part of the frame.
    let window: Option<Tile> = crate::argument_value("--reference-window").map(|v| {
        let c: Vec<u32> = v.split(',').map(|x| x.parse().unwrap()).collect();
        (c[0], c[1], c[2], c[3])
    });

    let scene = crate::selected_scene();
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
        tonemapper: crate::display_settings().0,
        bloom: crate::display_settings().1,
        camera: crate::scene_camera(&scene).renderer_camera(),
        exposure: 2.,
        auto_exposure: false,
        sky: crate::scene_sky(&scene),
        sampling: SamplingParams {
            max_samples_per_pixel: PASS << 24,
            num_samples_per_pixel: PASS,
            num_bounces: crate::selected_bounces(),
            denoising_enabled: false,
            ..Default::default()
        },
        viewport_size: (width, height),
    };
    params.camera.aperture = 0.;
    if let Some((position, direction, vfov)) = crate::camera_argument() {
        params.camera.eye_pos = position;
        params.camera.eye_dir = direction;
        params.camera.vfov = crate::raytracer::Angle::degrees(vfov);
    }
    let mut tracer = Raytracer::new(&device, &queue, &config, &scene, &params).unwrap();
    tracer.set_skip_filter(true);

    let tiles: Vec<Tile> = (0..height.div_ceil(tile_size))
        .flat_map(|ty| {
            (0..width.div_ceil(tile_size)).map(move |tx| {
                let (x, y) = (tx * tile_size, ty * tile_size);
                (x, y, tile_size.min(width - x), tile_size.min(height - y))
            })
        })
        .filter(|t| {
            window.is_none_or(|w| t.0 >= w.0 && t.1 >= w.1 && t.0 < w.0 + w.2 && t.1 < w.1 + w.3)
        })
        .collect();
    let pixels = (width * height) as usize;
    let groups = if uniform.is_some() { 1 } else { GROUPS };
    // Per group: the sum of every sample of each pixel; passes per group per tile.
    let mut sums = vec![vec![0f32; pixels * 3]; groups];
    let mut passes = vec![0u32; tiles.len()];
    let start = Instant::now();
    // Pixel-integrated albedo and normal (6 floats per pixel): each readback holds the
    // means over the last dispatch's samples (REFERENCE_GUIDES); summed over readbacks.
    let mut guides = vec![0f32; pixels * 6];
    let mut guide_reads = vec![0u32; tiles.len()];
    let mut add = |tracer: &Raytracer, sums: &mut Vec<f32>, active: &[usize], count: u32| {
        // read_radiance divides by the samples of the last tile traced: `count` passes.
        let mean = tracer.read_radiance(&device, &queue);
        let gbuffer = tracer.read_gbuffer(&device, &queue);
        let scale = (count * PASS) as f32;
        assert!(
            mean.iter().any(|&v| v > 0.),
            "the GPU returned an empty image (a dispatch was reset?)"
        );
        for &t in active {
            let (x, y, w, h) = tiles[t];
            guide_reads[t] += 1;
            for row in y..y + h {
                let from = ((row * width + x) * 3) as usize;
                for i in from..from + (w * 3) as usize {
                    sums[i] += mean[i] * scale;
                }
                for p in (row * width + x) as usize..(row * width + x + w) as usize {
                    for c in 0..3 {
                        guides[p * 6 + c] += gbuffer[p * 12 + c];
                        guides[p * 6 + 3 + c] += gbuffer[p * 12 + 4 + c];
                    }
                }
            }
        }
    };
    let mut rounds = 0;
    let mut worst = 0f32;
    let mut exposure = 0f32;
    if let Some(n) = uniform {
        let all: Vec<usize> = (0..tiles.len()).collect();
        for &t in &all {
            trace(&device, &queue, &mut tracer, tiles[t], n / PASS);
            passes[t] = n / PASS;
        }
        tracer.set_trace_region(None);
        add(&tracer, &mut sums[0], &all, n / PASS);
    } else {
        let mut active: Vec<usize> = (0..tiles.len()).collect();
        let mut count = ROUND;
        // Seconds per tile pass, measured.
        let mut pass_seconds;
        loop {
            let round = Instant::now();
            for group in sums.iter_mut() {
                for &t in &active {
                    trace(&device, &queue, &mut tracer, tiles[t], count);
                }
                // Readbacks cover the trace region's rows only: read the whole frame.
                tracer.set_trace_region(None);
                add(&tracer, group, &active, count);
            }
            for &t in &active {
                passes[t] += count;
            }
            rounds += 1;
            pass_seconds = round.elapsed().as_secs_f64()
                / (GROUPS as f64 * active.len() as f64 * f64::from(count));
            // Per pixel: the standard error of the mean luminance over the groups, in
            // log1p(exposure * x) units; per tile: its 95th percentile.
            if exposure == 0. {
                let traced: u32 = tiles.iter().map(|t| t.2 * t.3).sum();
                let mean = (0..pixels)
                    .map(|p| {
                        sums.iter()
                            .map(|g| luminance(&g[p * 3..p * 3 + 3]))
                            .sum::<f32>()
                            / (GROUPS as u32 * PASS * count) as f32
                    })
                    .sum::<f32>()
                    / traced as f32;
                exposure = 0.18 / mean.max(1e-8);
            }
            let mut errors: Vec<(f32, usize)> = tiles
                .iter()
                .enumerate()
                .map(|(t, &(x, y, w, h))| {
                    let n = (passes[t] * PASS) as f32;
                    let mut e: Vec<f32> = (y..y + h)
                        .flat_map(|row| (x..x + w).map(move |col| (row * width + col) as usize * 3))
                        .map(|p| {
                            let l: [f32; GROUPS] =
                                std::array::from_fn(|g| luminance(&sums[g][p..p + 3]) / n);
                            let mean = l.iter().sum::<f32>() / GROUPS as f32;
                            let variance = l.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                                / (GROUPS - 1) as f32;
                            exposure * (variance / GROUPS as f32).sqrt() / (1. + exposure * mean)
                        })
                        .collect();
                    e.sort_by(f32::total_cmp);
                    (e[e.len() * 95 / 100], t)
                })
                .collect();
            errors.sort_by(|a, b| b.0.total_cmp(&a.0));
            worst = errors[0].0;
            let left = seconds - start.elapsed().as_secs_f64();
            eprintln!(
                "round {rounds}: {} tiles x {} spp, worst tile error {worst:.4}, median {:.4}, {left:.0} s left",
                active.len(),
                GROUPS as u32 * count * PASS,
                errors[errors.len() / 2].0
            );
            // Next: the tiles above the target and within 0.6 of the worst, as many as
            // fit in the time left; few tiles get more passes (at most doubling theirs).
            active = errors
                .iter()
                .take_while(|e| e.0 > target.max(0.6 * worst))
                .map(|e| e.1)
                .collect();
            let fit = (left / (GROUPS as f64 * pass_seconds)) as usize;
            active.truncate(fit / ROUND as usize);
            if active.is_empty() {
                break;
            }
            let least = active.iter().map(|&t| passes[t]).min().unwrap();
            count = (((tiles.len() / 4).max(6) / active.len()) as u32)
                .min(least)
                .min((fit / active.len()) as u32)
                .max(ROUND);
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let mut bytes = Vec::with_capacity(groups * pixels * 12);
    let mut spp = vec![0f32; pixels];
    for (t, &(x, y, w, h)) in tiles.iter().enumerate() {
        for row in y..y + h {
            for col in x..x + w {
                spp[(row * width + col) as usize] = (passes[t] * PASS) as f32 * groups as f32;
            }
        }
    }
    for group in &sums {
        for (p, v) in group.iter().enumerate() {
            bytes.extend((v * groups as f32 / spp[p / 3].max(1.)).to_le_bytes());
        }
    }
    std::fs::write(output.join("groups.f32"), bytes).unwrap();
    std::fs::write(output.join("spp.f32"), bytemuck::cast_slice(&spp)).unwrap();
    drop(add);
    for (t, &(x, y, w, h)) in tiles.iter().enumerate() {
        for row in y..y + h {
            for p in (row * width + x) as usize..(row * width + x + w) as usize {
                for v in &mut guides[p * 6..p * 6 + 6] {
                    *v /= guide_reads[t] as f32;
                }
            }
        }
    }
    std::fs::write(output.join("guides.f32"), bytemuck::cast_slice(&guides)).unwrap();
    let traced = spp.iter().filter(|&&v| v > 0.).count();
    let mean_spp = spp.iter().sum::<f32>() / traced as f32;
    let (least, most) = spp
        .iter()
        .filter(|&&v| v > 0.)
        .fold((f32::MAX, 0f32), |a, &v| (a.0.min(v), a.1.max(v)));
    std::fs::write(
        output.join("meta.json"),
        format!(
            "{{\"width\":{width},\"height\":{height},\"groups\":{groups},\"seconds\":{elapsed:.1},\"mean_spp\":{mean_spp:.0},\"min_spp\":{least},\"max_spp\":{most},\"rounds\":{rounds},\"worst_tile_error\":{worst},\"target_error\":{target},\"exposure\":{exposure},\"bounces\":{}}}",
            params.sampling.num_bounces
        ),
    )
    .unwrap();
    eprintln!(
        "Reference {width}x{height}: {mean_spp:.0} spp on average ({least} to {most}), {elapsed:.0} s; saved {}",
        output.display()
    );
}
