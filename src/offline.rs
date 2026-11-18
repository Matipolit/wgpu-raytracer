//! Deterministic offscreen renders for inspecting imported scenes without a window.
#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]
use crate::{Raytracer, RenderParams, SamplingParams};
use std::time::Instant;

pub fn run() {
    pollster::block_on(render());
}
async fn render() {
    let path = crate::argument_value("--render-output").unwrap();
    let samples: u32 = crate::argument_value("--samples").map_or(32, |s| {
        s.parse().expect("--samples must be a positive integer")
    });
    assert!(samples > 0, "--samples must be positive");
    let dimensions = crate::argument_value("--size").unwrap_or_else(|| "800x600".into());
    let (width, height) = dimensions
        .split_once('x')
        .expect("--size must be WIDTHxHEIGHT");
    let (width, height): (u32, u32) = (width.parse().unwrap(), height.parse().unwrap());
    let scene = crate::selected_scene();
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let info = adapter.get_info();
    let required = crate::raytracer::gpu_textures::device_requirements(&adapter);
    let features =
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    let timestamps = adapter.features().contains(features);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: (if timestamps {
                features
            } else {
                wgpu::Features::empty()
            }) | crate::experiment_features(&adapter)
                | required.0,
            required_limits: required.1,
            experimental_features: crate::raytracer::gpu_textures::experimental_features(),
            ..Default::default()
        })
        .await
        .unwrap();
    let mut params = RenderParams {
        tonemapper: crate::display_settings().0,
        bloom: crate::display_settings().1,
        camera: crate::scene_camera(&scene).renderer_camera(),
        // An explicit --exposure keeps the historical fixed exposure; otherwise
        // automatic exposure (--exposure-compensation scales its result).
        exposure: crate::argument_value("--exposure").map_or_else(
            || {
                crate::argument_value("--exposure-compensation")
                    .map_or(crate::AUTO_EXPOSURE_COMPENSATION, |s| {
                        s.parse().expect("invalid exposure compensation")
                    })
            },
            |s| s.parse().expect("invalid exposure"),
        ),
        auto_exposure: crate::argument_value("--exposure").is_none(),
        sky: crate::scene_sky(&scene),
        sampling: SamplingParams {
            max_samples_per_pixel: samples,
            num_bounces: crate::selected_bounces(),
            ..Default::default()
        },
        viewport_size: (width, height),
    };
    params.sampling.denoising_enabled = !std::env::args().any(|a| a == "--no-denoise")
        && !crate::raytracer::upscale::joint_requested();
    // As the app (main.rs): a temporal denoiser or the joint network implies ReSTIR.
    params.sampling.restir = (crate::has_flag("--restir")
        || crate::has_flag("--temporal-model")
        || crate::raytracer::upscale::joint_requested())
        && !crate::has_flag("--no-restir");
    if let Some((position, direction, vfov)) = crate::camera_argument() {
        params.camera.eye_pos = position;
        params.camera.eye_dir = direction;
        params.camera.vfov = crate::raytracer::Angle::degrees(vfov);
    }
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
    let build = Instant::now();
    let mut tracer = Raytracer::new(&device, &queue, &config, &scene, &params).unwrap();
    tracer.calibrate(&device, &queue, &crate::adapter_key(&info));
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offline image"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let query = timestamps.then(|| {
        device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        })
    });
    let resolve = timestamps.then(|| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    });
    let timer = timestamps.then(|| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    });
    let setup = build.elapsed();
    let start = Instant::now();
    let mut gpu_ms = Vec::new();
    let (output_size, render_size) = tracer.sizes();
    let mut stability = crate::stability::Stability::from_args(output_size, render_size);
    // `--eval-velocity X,Y,Z` (scene units per frame) and `--eval-yaw DEGREES` per frame.
    let velocity = crate::argument_value("--eval-velocity").map(|v| {
        let c: Vec<f32> = v.split(',').map(|x| x.parse().unwrap()).collect();
        glm::vec3(c[0], c[1], c[2])
    });
    let yaw: f32 = crate::argument_value("--eval-yaw").map_or(0., |v| v.parse().unwrap());
    // `--raw-frames FILE`: every frame is a fresh 1-spp image (no accumulation), as the
    // real-time renderer traces it; after `--raw-warmup` frames (default 32), every
    // `--raw-every`-th (default 32) is appended to FILE as f32 rgb. Frames that far
    // apart are nearly independent of each other (ReSTIR history is capped at 20).
    let raw_frames = crate::argument_value("--raw-frames");
    let number = |name: &str, default: u32| {
        crate::argument_value(name).map_or(default, |v| v.parse().expect("a number"))
    };
    let (raw_warmup, raw_every) = (number("--raw-warmup", 32), number("--raw-every", 32));
    let mut raw = Vec::new();
    for frame in 0..samples {
        if raw_frames.is_some() {
            tracer.reset_accumulation();
        }
        if frame > 0 && (velocity.is_some() || yaw != 0.) {
            params.camera.eye_pos += velocity.unwrap_or_default();
            params.camera.eye_dir =
                glm::rotate_vec3(&params.camera.eye_dir, yaw.to_radians(), &params.camera.up);
            tracer.set_render_params(&device, &queue, &params).unwrap();
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        if let Some(query) = &query {
            encoder.write_timestamp(query, 0);
        }
        tracer.accumulate_frame(&queue, &mut encoder);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("offline display"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            tracer.render_frame(&mut pass);
        }
        if let Some(query) = &query {
            encoder.write_timestamp(query, 1);
            encoder.resolve_query_set(query, 0..2, resolve.as_ref().unwrap(), 0);
            encoder.copy_buffer_to_buffer(
                resolve.as_ref().unwrap(),
                0,
                timer.as_ref().unwrap(),
                0,
                16,
            );
        }
        queue.submit([encoder.finish()]);
        tracer.after_submit(&device);
        if let Some(timer) = &timer {
            timer
                .slice(..)
                .map_async(wgpu::MapMode::Read, |r| r.unwrap());
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        if let Some(timer) = &timer {
            {
                let data = timer.slice(..).get_mapped_range().unwrap();
                let ticks: &[u64] = bytemuck::cast_slice(&data);
                // First frame includes transfers and driver pipeline compilation.
                if frame > 0 {
                    gpu_ms.push(
                        (ticks[1] - ticks[0]) as f64 * f64::from(queue.get_timestamp_period())
                            / 1e6,
                    );
                }
            }
            timer.unmap();
        }
        if raw_frames.is_some() && frame >= raw_warmup && (frame - raw_warmup) % raw_every == 0 {
            raw.extend(tracer.read_radiance(&device, &queue));
        }
        if let Some(stability) = &mut stability {
            stability.frame(
                frame,
                &tracer.read_display(&device, &queue),
                &tracer.read_gbuffer(&device, &queue),
            );
        }
        if samples >= 512 && (frame + 1) % 256 == 0 {
            eprintln!(
                "Progress: {} / {samples} spp; {:.1}s elapsed",
                frame + 1,
                start.elapsed().as_secs_f64()
            );
        }
    }
    if let Some(path) = raw_frames {
        std::fs::write(path, bytemuck::cast_slice(&raw)).unwrap();
    }
    if let Some(raw) = crate::argument_value("--raw-output") {
        tracer.save_raw(&device, &queue, &raw, samples);
    }
    // Rasterized-visibility experiment: traced G-buffer first, then the raster one.
    if let Some(path) = crate::argument_value("--gbuffer-output") {
        std::fs::write(
            path,
            bytemuck::cast_slice(&tracer.read_gbuffer(&device, &queue)),
        )
        .unwrap();
    }
    if crate::has_flag("--visibility") {
        let [raster, resolve] = tracer.benchmark_visibility(&device, &queue);
        println!(
            "visibility: raster {raster:.3} ms, resolve {resolve:.3} ms, total {:.3} ms",
            raster + resolve
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        tracer.encode_visibility(&mut encoder);
        queue.submit([encoder.finish()]);
        if let Some(path) = crate::argument_value("--visibility-output") {
            std::fs::write(
                path,
                bytemuck::cast_slice(&tracer.read_gbuffer(&device, &queue)),
            )
            .unwrap();
        }
    }
    if let Some(report) = crate::argument_value("--filter-report") {
        let ms = tracer.benchmark_filter(&device, &queue);
        std::fs::write(
            report,
            format!(
                "{{\"gpu\":{:?},\"width\":{width},\"height\":{height},\"filter_p95_ms\":{ms}}}",
                info.name
            ),
        )
        .unwrap();
    }
    let elapsed = start.elapsed();
    let stride = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(stride) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: None,
            },
        },
        texture.size(),
    );
    queue.submit([encoder.finish()]);
    output
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let rgba = {
        let mapped = output.slice(..).get_mapped_range().unwrap();
        mapped
            .chunks_exact(stride as usize)
            .flat_map(|row| row[..width as usize * 4].iter().copied())
            .collect::<Vec<_>>()
    };
    output.unmap();
    image::save_buffer(&path, &rgba, width, height, image::ColorType::Rgba8).unwrap();
    gpu_ms.sort_by(f64::total_cmp);
    if let Some(report) = crate::argument_value("--timing-output") {
        assert!(
            !gpu_ms.is_empty(),
            "timing output requires GPU timestamps and at least 2 samples"
        );
        let total: f64 = gpu_ms.iter().sum();
        std::fs::write(report, format!(
            "{{\"samples\":{samples},\"warmed_samples\":{},\"gpu_total_ms\":{total},\"gpu_median_ms\":{},\"gpu_p95_ms\":{}}}",
            gpu_ms.len(), gpu_ms[gpu_ms.len()/2], gpu_ms[(gpu_ms.len()*95/100).min(gpu_ms.len()-1)]
        )).unwrap();
    }
    // Scene key: log-average luminance of the displayed radiance (as auto exposure sees it).
    let shown = tracer.read_filtered(&device, &queue);
    let logs: Vec<f64> = shown
        .chunks_exact(3)
        .map(|c| {
            (0.2126 * f64::from(c[0]) + 0.7152 * f64::from(c[1]) + 0.0722 * f64::from(c[2]))
                .max(1e-4)
                .ln()
        })
        .collect();
    eprintln!(
        "Scene key (log-average luminance): {:.4}",
        (logs.iter().sum::<f64>() / logs.len().max(1) as f64).exp()
    );
    if let Some(times) = tracer.pass_times() {
        eprintln!("{times}");
    }
    if let Some(stability) = &stability {
        stability.dump();
        println!("{}", stability.report());
    }
    eprintln!("Auto-tune: ended on {}", tracer.trace_variant());
    eprintln!(
        "Rendered {width}x{height}, {samples} spp on {} ({:?}); setup {:.2}s, render {:.2}s; warmed median GPU {:?} ms/sample; saved {path}",
        info.name,
        info.backend,
        setup.as_secs_f64(),
        elapsed.as_secs_f64(),
        gpu_ms.get(gpu_ms.len() / 2)
    );
}
