//! Rasterized primary visibility (`--visibility`; see visibility.wgsl).
//! A depth-tested raster pass writes the index of the closest triangle per pixel;
//! a compute pass resolves it into the G-buffer with the tracer's own functions.
//! Vertices are pulled from the tracer's mesh buffer, so no geometry is duplicated.
//! Triangles are drawn in three groups so early depth testing survives where it can:
//! double-sided opaque (no culling), single-sided opaque (hardware back-face
//! culling) and alpha-tested (fragment alpha test and back-face rule, `discard`).
#![allow(clippy::cast_possible_truncation)]
use super::mesh::Mesh;
use wgpu::util::DeviceExt;

const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const CLUSTER: u32 = 256;

/// GPU cluster record (visibility.wgsl `VisCluster`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCluster {
    lo: [f32; 3],
    first_index: u32,
    hi: [f32; 3],
    index_count: u32,
    triangle_offset: u32,
    pad: [u32; 3],
}

/// GPU-driven two-pass occlusion culling (needs INDIRECT_FIRST_INSTANCE). Opt-in:
/// it culled little on the benchmark views (Bistro: ~3% of in-frustum clusters).
struct Occlusion {
    count: u32,
    /// Per triangle group: [first, end) cluster indices (clusters are group-ordered).
    group_ranges: [(u32, u32); 3],
    clusters: wgpu::Buffer,
    /// Indirect draw arguments of pass 1 and pass 2, 20 bytes per cluster.
    args: [wgpu::Buffer; 2],
    visible: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    group: wgpu::BindGroup,
    cull1: wgpu::ComputePipeline,
    hiz: wgpu::ComputePipeline,
    cull2: wgpu::ComputePipeline,
}

impl Occlusion {
    fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("visibility culling layout"),
            entries: &[
                frame_entry(wgpu::ShaderStages::COMPUTE),
                storage(8, true),
                storage(9, false),
                storage(10, false),
                storage(11, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage(13, false),
            ],
        })
    }

    /// Bindings that depend on the frame size (depth target, tile depths).
    fn bind(
        &mut self,
        device: &wgpu::Device,
        frame: &wgpu::Buffer,
        depth: &wgpu::TextureView,
        size: (u32, u32),
    ) {
        let tiles = u64::from(size.0.div_ceil(16)) * u64::from(size.1.div_ceil(16));
        let hiz = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility tile depths"),
            size: 4 * tiles.max(1),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        self.group = Self::group(
            device,
            &self.layout,
            frame,
            &self.clusters,
            &self.args,
            &self.visible,
            depth,
            &hiz,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        frame: &wgpu::Buffer,
        clusters: &wgpu::Buffer,
        args: &[wgpu::Buffer; 2],
        visible: &wgpu::Buffer,
        depth: &wgpu::TextureView,
        hiz: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        fn buffer(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visibility culling group"),
            layout,
            entries: &[
                buffer(2, frame),
                buffer(8, clusters),
                buffer(9, &args[0]),
                buffer(10, &args[1]),
                buffer(11, visible),
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                buffer(13, hiz),
            ],
        })
    }
}

pub struct Visibility {
    /// Per group: (first index, index count, offset in the triangle list).
    ranges: [(u32, u32, u32); 3],
    /// Per group: bounding boxes of consecutive runs of CLUSTER triangles (BVH order,
    /// so spatially compact), for frustum culling.
    clusters: [Vec<([f32; 3], [f32; 3])>; 3],
    indices: wgpu::Buffer,
    /// Positions only (12 bytes per vertex), fetched by the input assembler.
    positions: wgpu::Buffer,
    pipelines: [wgpu::RenderPipeline; 3],
    resolve: wgpu::ComputePipeline,
    raster_group: wgpu::BindGroup,
    resolve_layout: wgpu::BindGroupLayout,
    resolve_group: wgpu::BindGroup,
    target: wgpu::TextureView,
    depth: wgpu::TextureView,
    size: (u32, u32),
    /// VisFrame uniform: jitter.xy, width, height.
    frame: wgpu::Buffer,
    occlusion: Option<Occlusion>,
}

fn frame_entry(visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 2,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn raster_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("visibility raster layout"),
        entries: &[
            frame_entry(wgpu::ShaderStages::VERTEX),
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

pub fn resolve_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("visibility resolve layout"),
        entries: &[
            frame_entry(wgpu::ShaderStages::COMPUTE),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

/// Extra stages that read the parameter and scene bind groups.
pub fn stages(enabled: bool) -> wgpu::ShaderStages {
    if enabled {
        wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT
    } else {
        wgpu::ShaderStages::empty()
    }
}

impl Visibility {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        mesh: &Mesh,
        shader: &wgpu::ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
        raster_layout: &wgpu::BindGroupLayout,
        resolve_layout: wgpu::BindGroupLayout,
        resolve: wgpu::ComputePipeline,
        constants: &[(&str, f64)],
        size: (u32, u32),
        parameter_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let lists = mesh.visibility_triangles();
        let mut all = Vec::new();
        let mut start = 0;
        let mut indices: Vec<u32> = Vec::new();
        let ranges = lists.each_ref().map(|list| {
            all.extend_from_slice(list);
            indices.extend(
                list.iter()
                    .flat_map(|&t| mesh.triangles[t as usize].vertices),
            );
            let range = (start * 3, list.len() as u32 * 3, start);
            start += list.len() as u32;
            range
        });
        let clusters: [Vec<([f32; 3], [f32; 3])>; 3] = lists.each_ref().map(|list| {
            list.chunks(CLUSTER as usize)
                .map(|chunk| {
                    let mut lo = [f32::MAX; 3];
                    let mut hi = [f32::MIN; 3];
                    for &t in chunk {
                        for &v in &mesh.triangles[t as usize].vertices {
                            let p = mesh.vertices[v as usize].position;
                            for k in 0..3 {
                                lo[k] = lo[k].min(p[k]);
                                hi[k] = hi[k].max(p[k]);
                            }
                        }
                    }
                    (lo, hi)
                })
                .collect()
        });
        let positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.position).collect();
        let positions = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("visibility positions"),
            contents: bytemuck::cast_slice(if positions.is_empty() {
                &[[0f32; 3]][..]
            } else {
                &positions
            }),
            usage: wgpu::BufferUsages::VERTEX,
        });
        if indices.is_empty() {
            indices.extend([0, 0, 0]);
        }
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("visibility indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        if all.is_empty() {
            all.push(0);
        }
        eprintln!(
            "Visibility raster: {} double-sided, {} single-sided, {} alpha-tested triangles",
            lists[0].len(),
            lists[1].len(),
            lists[2].len()
        );
        let list_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("visibility triangle lists"),
            contents: bytemuck::cast_slice(&all),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let frame = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("visibility frame"),
            contents: bytemuck::cast_slice(&[0.5f32, 0.5, size.0 as f32, size.1 as f32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let raster_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visibility raster group"),
            layout: raster_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: list_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frame.as_entire_binding(),
                },
            ],
        });
        // Front faces are counter-clockwise in NDC for the tracer's convention
        // (single-sided triangles face the viewer when dot(ray, cross(e1, e2)) < 0).
        let front_face = if std::env::var("RT_VIS_CW").is_ok() {
            wgpu::FrontFace::Cw
        } else {
            wgpu::FrontFace::Ccw
        };
        let pipeline = |fragment: &str, cull_mode| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("visibility raster"),
                layout: Some(pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vsVisibility"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants,
                        ..Default::default()
                    },
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: 12,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some(fragment),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants,
                        ..Default::default()
                    },
                    targets: &[Some(wgpu::ColorTargetState {
                        format: TARGET_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face,
                    cull_mode,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipelines = [
            pipeline("fsVisibility", None),
            pipeline("fsVisibility", Some(wgpu::Face::Back)),
            pipeline("fsVisibilityTested", None),
        ];
        let (target, depth, resolve_group) = Self::targets(device, &resolve_layout, &frame, size);
        // GPU-driven occlusion culling: per-draw triangle offsets come from the indirect
        // first instance, which needs INDIRECT_FIRST_INSTANCE.
        let occlusion = device
            .features()
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
            .then(|| {
                let mut records = Vec::new();
                let group_ranges = std::array::from_fn(|g| {
                    let begin = records.len() as u32;
                    let (first, _, offset) = ranges[g];
                    for (c, &(lo, hi)) in clusters[g].iter().enumerate() {
                        let start = c as u32 * CLUSTER;
                        let count = (lists[g].len() as u32 - start).min(CLUSTER);
                        records.push(GpuCluster {
                            lo,
                            first_index: first + start * 3,
                            hi,
                            index_count: count * 3,
                            triangle_offset: offset + start,
                            pad: [0; 3],
                        });
                    }
                    (begin, records.len() as u32)
                });
                let count = records.len() as u32;
                if records.is_empty() {
                    records.push(GpuCluster {
                        lo: [0.; 3],
                        first_index: 0,
                        hi: [0.; 3],
                        index_count: 0,
                        triangle_offset: 0,
                        pad: [0; 3],
                    });
                }
                let clusters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("visibility clusters"),
                    contents: bytemuck::cast_slice(&records),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let args = std::array::from_fn(|_| {
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("visibility draw arguments"),
                        size: 20 * u64::from(count.max(1)),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                        mapped_at_creation: false,
                    })
                });
                let visible = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("visibility visible clusters"),
                    contents: bytemuck::cast_slice(&vec![1u32; records.len()]),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let layout = Occlusion::layout(device);
                let pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("visibility culling layout"),
                        bind_group_layouts: &[
                            None,
                            None,
                            Some(parameter_layout),
                            None,
                            None,
                            None,
                            Some(&layout),
                        ],
                        immediate_size: 0,
                    });
                let pipeline = |entry: &str| {
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(entry),
                        layout: Some(&pipeline_layout),
                        module: shader,
                        entry_point: Some(entry),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants,
                            ..Default::default()
                        },
                        cache: None,
                    })
                };
                let hiz = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 4,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                let group = Occlusion::group(
                    device, &layout, &frame, &clusters, &args, &visible, &depth, &hiz,
                );
                let mut occlusion = Occlusion {
                    count,
                    group_ranges,
                    clusters,
                    args,
                    visible,
                    layout,
                    group,
                    cull1: pipeline("csVisCull1"),
                    hiz: pipeline("csVisHiz"),
                    cull2: pipeline("csVisCull2"),
                };
                occlusion.bind(device, &frame, &depth, size);
                occlusion
            });
        Self {
            ranges,
            clusters,
            indices,
            positions,
            pipelines,
            resolve,
            raster_group,
            resolve_layout,
            resolve_group,
            target,
            depth,
            size,
            frame,
            occlusion,
        }
    }

    fn targets(
        device: &wgpu::Device,
        resolve_layout: &wgpu::BindGroupLayout,
        frame: &wgpu::Buffer,
        (width, height): (u32, u32),
    ) -> (wgpu::TextureView, wgpu::TextureView, wgpu::BindGroup) {
        let texture = |label, format, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let target = texture(
            "visibility buffer",
            TARGET_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let depth = texture(
            "visibility depth",
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visibility resolve group"),
            layout: resolve_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&target),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frame.as_entire_binding(),
                },
            ],
        });
        (target, depth, group)
    }

    pub fn resize(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        (self.target, self.depth, self.resolve_group) =
            Self::targets(device, &self.resolve_layout, &self.frame, size);
        self.size = size;
        if let Some(occlusion) = &mut self.occlusion {
            occlusion.bind(device, &self.frame, &self.depth, size);
        }
    }

    /// Render pass over the visibility targets with the raster bindings set.
    fn begin<'e>(
        &'e self,
        encoder: &'e mut wgpu::CommandEncoder,
        parameters: &'e wgpu::BindGroup,
        scene: &'e wgpu::BindGroup,
        clear: bool,
    ) -> wgpu::RenderPass<'e> {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("visibility raster pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if clear {
                        wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(u32::MAX),
                            g: 0.,
                            b: 0.,
                            a: 0.,
                        })
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth,
                depth_ops: Some(wgpu::Operations {
                    load: if clear {
                        wgpu::LoadOp::Clear(0.)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(2, parameters, &[]);
        pass.set_bind_group(3, scene, &[]);
        pass.set_bind_group(6, &self.raster_group, &[]);
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_vertex_buffer(0, self.positions.slice(..));
        pass
    }

    /// Raster pass; `parameters` and `scene` are the bind groups 2 and 3.
    pub fn raster(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        parameters: &wgpu::BindGroup,
        scene: &wgpu::BindGroup,
        planes: &[[f32; 4]; 4],
    ) {
        // Opt-in (RT_VIS_OCCLUSION=1): on the benchmark views the camera-facing geometry
        // is mostly visible, and the GPU-driven passes cost more than they cull.
        if let Some(occlusion) = self
            .occlusion
            .as_ref()
            .filter(|_| std::env::var("RT_VIS_OCCLUSION").is_ok_and(|v| v == "1"))
        {
            self.raster_occluded(encoder, parameters, scene, occlusion);
            return;
        }
        let mut pass = self.begin(encoder, parameters, scene, true);
        // RT_VIS_LISTS=abc (0/1 per group) draws only some groups (profiling).
        let mask = std::env::var("RT_VIS_LISTS").unwrap_or_else(|_| "111".into());
        let cull = std::env::var("RT_VIS_NO_CULL").is_err();
        for (((pipeline, &(first, count, offset)), clusters), on) in self
            .pipelines
            .iter()
            .zip(&self.ranges)
            .zip(&self.clusters)
            .zip(mask.chars())
        {
            if count == 0 || on != '1' {
                continue;
            }
            pass.set_pipeline(pipeline);
            if !cull {
                pass.draw_indexed(first..first + count, 0, offset..offset + 1);
                continue;
            }
            // Merge runs of visible clusters into single draws.
            let triangles = count / 3;
            let mut run: Option<u32> = None;
            for (c, &(lo, hi)) in clusters.iter().enumerate() {
                let visible = planes.iter().all(|p| {
                    let x = if p[0] >= 0. { hi[0] } else { lo[0] };
                    let y = if p[1] >= 0. { hi[1] } else { lo[1] };
                    let z = if p[2] >= 0. { hi[2] } else { lo[2] };
                    p[0] * x + p[1] * y + p[2] * z + p[3] >= 0.
                });
                let c = c as u32;
                match (visible, run) {
                    (true, None) => run = Some(c),
                    (false, Some(start)) => {
                        Self::draw_clusters(&mut pass, first, offset, start, c, triangles);
                        run = None;
                    }
                    _ => {}
                }
            }
            if let Some(start) = run {
                Self::draw_clusters(
                    &mut pass,
                    first,
                    offset,
                    start,
                    clusters.len() as u32,
                    triangles,
                );
            }
        }
    }

    /// Two-pass GPU occlusion culling (see visibility.wgsl): clusters visible last frame,
    /// a depth pyramid of them, then every other cluster that may be in front of it.
    fn raster_occluded(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        parameters: &wgpu::BindGroup,
        scene: &wgpu::BindGroup,
        occlusion: &Occlusion,
    ) {
        let cull = |encoder: &mut wgpu::CommandEncoder,
                    pipelines: &[(&wgpu::ComputePipeline, (u32, u32))]| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("visibility culling"),
                timestamp_writes: None,
            });
            pass.set_bind_group(2, parameters, &[]);
            pass.set_bind_group(6, &occlusion.group, &[]);
            for (pipeline, (x, y)) in pipelines {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(*x, *y, 1);
            }
        };
        let groups = occlusion.count.div_ceil(64);
        cull(encoder, &[(&occlusion.cull1, (groups, 1))]);
        for (args, clear) in [(&occlusion.args[0], true), (&occlusion.args[1], false)] {
            if !clear {
                let tiles = (self.size.0.div_ceil(16), self.size.1.div_ceil(16));
                cull(
                    encoder,
                    &[(&occlusion.hiz, tiles), (&occlusion.cull2, (groups, 1))],
                );
            }
            let mut pass = self.begin(encoder, parameters, scene, clear);
            for (pipeline, &(first, end)) in self.pipelines.iter().zip(&occlusion.group_ranges) {
                if end > first {
                    pass.set_pipeline(pipeline);
                    pass.multi_draw_indexed_indirect(args, u64::from(first) * 20, end - first);
                }
            }
        }
    }

    fn draw_clusters(
        pass: &mut wgpu::RenderPass<'_>,
        first: u32,
        offset: u32,
        start: u32,
        end: u32,
        triangles: u32,
    ) {
        let begin = start * CLUSTER;
        let end = (end * CLUSTER).min(triangles);
        pass.draw_indexed(
            first + begin * 3..first + end * 3,
            0,
            offset + begin..offset + begin + 1,
        );
    }

    /// This frame's sub-pixel jitter (Halton 2, 3), shared by the raster and the
    /// tracer's primary rays; pixel centres when `jitter` is false.
    pub fn set_frame(&self, queue: &wgpu::Queue, frame: u32, jitter: bool) {
        fn halton(mut i: u32, base: u32) -> f32 {
            let (mut f, mut r) = (1., 0.);
            while i > 0 {
                f /= base as f32;
                r += f * (i % base) as f32;
                i /= base;
            }
            r
        }
        let j = if jitter {
            [halton(frame % 1024 + 1, 2), halton(frame % 1024 + 1, 3)]
        } else {
            [0.5, 0.5]
        };
        queue.write_buffer(
            &self.frame,
            0,
            bytemuck::cast_slice(&[j[0], j[1], self.size.0 as f32, self.size.1 as f32]),
        );
    }

    /// Group 6 for the tracing pass (visibility buffer + frame).
    pub fn trace_group(&self) -> &wgpu::BindGroup {
        &self.resolve_group
    }

    /// Resolve pass; groups 0-3 must already be the tracing groups.
    pub fn resolve(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_bind_group(6, &self.resolve_group, &[]);
        pass.set_pipeline(&self.resolve);
        pass.dispatch_workgroups(self.size.0.div_ceil(8), self.size.1.div_ceil(8), 1);
    }
}
