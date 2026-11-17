//! Hardware ray-query acceleration structures (`--hw-rt`; see hwrt.wgsl).
//! One BLAS with up to two triangle geometries (opaque double-sided triangles,
//! and triangles that need the shader), in a single-instance TLAS.
#![allow(clippy::cast_possible_truncation)]
use super::mesh::Mesh;
use wgpu::util::DeviceExt;

pub struct HwAccel {
    pub tlas: wgpu::Tlas,
    _blas: wgpu::Blas,
}

impl HwAccel {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, mesh: &Mesh) -> Self {
        let start = std::time::Instant::now();
        let positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.position).collect();
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BLAS vertices"),
            contents: bytemuck::cast_slice(&positions),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });
        let lists = mesh.hw_geometry_triangles();
        // Geometry order must match hwrt.wgsl: 0 = opaque, 1 = shader-tested. An
        // empty first list still needs a slot, so it gets one degenerate triangle.
        let indices: Vec<Vec<u32>> = lists
            .iter()
            .map(|list| {
                let mut out: Vec<u32> = list
                    .iter()
                    .flat_map(|&t| mesh.triangles[t as usize].vertices)
                    .collect();
                if out.is_empty() {
                    out = vec![0, 0, 0];
                }
                out
            })
            .collect();
        let index_buffers: Vec<wgpu::Buffer> = indices
            .iter()
            .map(|list| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("BLAS indices"),
                    contents: bytemuck::cast_slice(list),
                    usage: wgpu::BufferUsages::BLAS_INPUT,
                })
            })
            .collect();
        let sizes: Vec<wgpu::BlasTriangleGeometrySizeDescriptor> = indices
            .iter()
            .enumerate()
            .map(|(g, list)| wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: positions.len() as u32,
                index_format: Some(wgpu::IndexFormat::Uint32),
                index_count: Some(list.len() as u32),
                flags: if g == 0 {
                    wgpu::AccelerationStructureGeometryFlags::OPAQUE
                } else {
                    wgpu::AccelerationStructureGeometryFlags::empty()
                },
            })
            .collect();
        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("scene BLAS"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: sizes.clone(),
            },
        );
        let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("scene TLAS"),
            max_instances: 1,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        tlas[0] = Some(wgpu::TlasInstance::new(
            &blas,
            [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
            0,
            0xff,
        ));
        let geometry = sizes
            .iter()
            .zip(&index_buffers)
            .map(|(size, index_buffer)| wgpu::BlasTriangleGeometry {
                size,
                vertex_buffer: &vertices,
                first_vertex: 0,
                vertex_stride: 12,
                index_buffer: Some(index_buffer),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            })
            .collect();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.build_acceleration_structures(
            std::iter::once(&wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(geometry),
            }),
            std::iter::once(&tlas),
        );
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        eprintln!(
            "Hardware BLAS: {} opaque + {} shader-tested triangles, {:.2}s",
            lists[0].len(),
            lists[1].len(),
            start.elapsed().as_secs_f64()
        );
        Self { tlas, _blas: blas }
    }

    pub fn layout_entry(visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: 8,
            visibility,
            ty: wgpu::BindingType::AccelerationStructure {
                vertex_return: false,
            },
            count: None,
        }
    }
}
