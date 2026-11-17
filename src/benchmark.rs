//! Offscreen release benchmark: no vsync, fixed scene/camera, warmed pipelines.
use crate::{Raytracer, RenderParams, SamplingParams};
use std::time::Instant;

const WARMUP_FRAMES: usize = 24;
const MEASURED_FRAMES: usize = 96;

pub fn run() {
    pollster::block_on(run_async());
}

async fn run_async() {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let info = adapter.get_info();
    let features =
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    assert!(
        adapter.features().contains(features),
        "Benchmark requires GPU timestamp queries"
    );
    let required = crate::raytracer::gpu_textures::device_requirements(&adapter);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: features | crate::experiment_features(&adapter) | required.0,
            required_limits: required.1,
            experimental_features: crate::raytracer::gpu_textures::experimental_features(),
            ..Default::default()
        })
        .await
        .unwrap();
    println!(
        "{{\"ui\":\"egui\",\"renderer\":\"compute\",\"trace_workgroup\":{:?},\"adapter\":{:?},\"backend\":{:?},\"driver\":{:?},\"warmup_frames\":{WARMUP_FRAMES},\"measured_frames\":{MEASURED_FRAMES},\"cases\":[",
        crate::raytracer::trace_workgroup_size(),
        info.name,
        format!("{:?}", info.backend),
        info.driver_info
    );
    let scene = crate::selected_scene();
    let mut first = true;
    for (width, height) in [(800, 600), (2560, 1600)] {
        for converged in [false, true] {
            let mut camera = crate::scene_camera(&scene);
            let mut params = RenderParams {
                tonemapper: crate::display_settings().0,
                bloom: crate::display_settings().1,
                camera: camera.renderer_camera(),
                exposure: crate::scene_exposure(&scene),
                auto_exposure: false,
                sky: crate::scene_sky(&scene),
                sampling: SamplingParams {
                    num_samples_per_pixel: if converged { 8 } else { 1 },
                    max_samples_per_pixel: if converged { 128 } else { 512 },
                    num_bounces: crate::selected_bounces(),
                    ..Default::default()
                },
                viewport_size: (width, height),
            };
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
            let mut tracer = Raytracer::new(&device, &queue, &config, &scene, &params).unwrap();
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("benchmark output"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("frame timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut gui = crate::gui::Gui::new(&device, config.format);
            let mut gpu_ms = Vec::new();
            let mut cpu_ms = Vec::new();
            let mut ui_cpu_ms = Vec::new();
            let mut wall_ms = Vec::new();
            for frame in 0..WARMUP_FRAMES + MEASURED_FRAMES {
                let start = Instant::now();
                let ui_start = Instant::now();
                let output = gui.run(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(width as f32, height as f32),
                        )),
                        time: Some(frame as f64 / 60.),
                        predicted_dt: 1. / 60.,
                        ..Default::default()
                    },
                    &mut params,
                    &mut camera,
                    60.,
                    tracer.progress(),
                );
                let ui_prepare_ms = ui_start.elapsed().as_secs_f64() * 1000.;
                tracer.set_render_params(&device, &queue, &params).unwrap();
                let mut encoder = device.create_command_encoder(&Default::default());
                encoder.write_timestamp(&queries, 0);
                let upload_start = Instant::now();
                let ui_commands =
                    gui.prepare(&device, &queue, &mut encoder, output, [width, height]);
                let ui_upload_ms = upload_start.elapsed().as_secs_f64() * 1000.;
                tracer.accumulate_frame(&queue, &mut encoder);
                let ui_render_ms;
                {
                    let mut pass = encoder
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("benchmark display and UI"),
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
                        })
                        .forget_lifetime();
                    tracer.render_frame(&mut pass);
                    let ui_start = Instant::now();
                    gui.render(&mut pass);
                    ui_render_ms = ui_start.elapsed().as_secs_f64() * 1000.;
                }
                encoder.write_timestamp(&queries, 1);
                encoder.resolve_query_set(&queries, 0..2, &resolve, 0);
                encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, 16);
                queue.submit(ui_commands.into_iter().chain([encoder.finish()]));
                tracer.after_submit(&device);
                gui.free_textures();
                let cpu = start.elapsed().as_secs_f64() * 1000.;
                readback
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, |result| result.unwrap());
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let wall = start.elapsed().as_secs_f64() * 1000.;
                {
                    let mapped = readback.slice(..).get_mapped_range().unwrap();
                    let ticks: &[u64] = bytemuck::cast_slice(&mapped);
                    if frame >= WARMUP_FRAMES {
                        gpu_ms.push(
                            (ticks[1] - ticks[0]) as f64 * f64::from(queue.get_timestamp_period())
                                / 1e6,
                        );
                        cpu_ms.push(cpu);
                        ui_cpu_ms.push(ui_prepare_ms + ui_upload_ms + ui_render_ms);
                        wall_ms.push(wall);
                    }
                }
                readback.unmap();
            }
            if !first {
                println!(",");
            }
            first = false;
            println!(
                "{{\"width\":{width},\"height\":{height},\"state\":{:?},\"gpu_ms\":{},\"cpu_ms\":{},\"ui_cpu_ms\":{},\"wall_ms\":{}}}",
                if converged {
                    "converged"
                } else {
                    "accumulating"
                },
                stats(&mut gpu_ms),
                stats(&mut cpu_ms),
                stats(&mut ui_cpu_ms),
                stats(&mut wall_ms)
            );
        }
    }
    println!("]}}");
}

fn stats(samples: &mut [f64]) -> String {
    samples.sort_by(f64::total_cmp);
    format!(
        "{{\"median\":{:.6},\"p95\":{:.6}}}",
        samples[samples.len() / 2],
        samples[samples.len() * 95 / 100]
    )
}
