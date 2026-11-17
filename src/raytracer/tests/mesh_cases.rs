use super::*;
use crate::raytracer::mesh::{Mesh, Triangle};
use std::path::Path;

fn fixture() -> Mesh {
    Mesh::load(Path::new("tests/assets/triangles.gltf")).unwrap()
}

#[test]
fn gltf_applies_transforms_and_preserves_mirrored_winding() {
    let mesh = fixture();
    assert_eq!(mesh.triangles.len(), 2);
    assert_eq!(mesh.vertices[0].position, [-1., -1., -3.]);
    assert_eq!(mesh.vertices[3].position, [6., -1., -3.]);
    for triangle in &mesh.triangles {
        let [a, b, c] = triangle
            .vertices
            .map(|v| glm::Vec3::from(mesh.vertices[v as usize].position));
        let geometric = glm::normalize(&glm::cross(&(b - a), &(c - a)));
        let shading = glm::Vec3::from(mesh.vertices[triangle.vertices[0] as usize].normal);
        assert!(glm::dot(&geometric, &shading) > 0.999);
    }
    assert!(mesh.depth < 64);
}

#[test]
fn bvh_bounds_enclose_all_leaves_and_every_triangle_once() {
    let mesh = grid();
    let mut seen = vec![0; mesh.triangles.len()];
    fn visit(mesh: &Mesh, node: usize, seen: &mut [u32]) {
        let n = mesh.nodes[node];
        if n.count == 0 {
            for child in [node + 1, n.first as usize] {
                let c = mesh.nodes[child];
                for axis in 0..3 {
                    assert!(c.min[axis] >= n.min[axis]);
                    assert!(c.max[axis] <= n.max[axis]);
                }
                visit(mesh, child, seen);
            }
        } else {
            for i in n.first..n.first + n.count {
                seen[i as usize] += 1;
                for vertex in mesh.triangles[i as usize].vertices {
                    for axis in 0..3 {
                        let p = mesh.vertices[vertex as usize].position[axis];
                        assert!(p >= n.min[axis] && p <= n.max[axis]);
                    }
                }
            }
        }
    }
    visit(&mesh, 0, &mut seen);
    assert!(seen.iter().all(|&n| n == 1));
    assert!(mesh.depth < 64);
}

fn grid() -> Mesh {
    let mut mesh = fixture();
    let vertices = mesh.vertices[..3].to_vec();
    mesh.vertices.clear();
    mesh.triangles.clear();
    // Overlapping layers, nonuniform positions and triangles crossing box planes.
    for z in 0..3 {
        for y in 0..12 {
            for x in 0..16 {
                let offset = mesh.vertices.len() as u32;
                for mut vertex in vertices.clone() {
                    vertex.position[0] += x as f32 * 1.7 - 12.;
                    vertex.position[1] += y as f32 * 1.3 - 7.;
                    vertex.position[2] -= z as f32 * 0.8;
                    mesh.vertices.push(vertex);
                }
                mesh.triangles.push(Triangle {
                    vertices: [offset, offset + 1, offset + 2],
                    material: 0,
                });
            }
        }
    }
    mesh.build_bvh();
    mesh
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Query {
    origin: [f32; 4],
    direction: [f32; 4],
    hit: [f32; 4],
    brute: [f32; 4],
    normal: [f32; 4],
    brute_normal: [f32; 4],
}

const QUERY_SHADER: &str = r"
struct Query {origin:vec4<f32>,direction:vec4<f32>,hit:vec4<f32>,brute:vec4<f32>,normal:vec4<f32>,bruteNormal:vec4<f32>}
@group(0) @binding(1) var<storage,read_write> queries:array<Query>;
@compute @workgroup_size(64)
fn csVndf(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&queries) {return;}
    let nv=queries[id.x].origin.x;let rough=queries[id.x].origin.y;let metal=queries[id.x].origin.z;
    let alpha=rough*rough;let view=vec3(sqrt(max(0.,1.-nv*nv)),0.,nv);
    let f0=mix(vec3(0.04),vec3(0.8),metal);
    let surface=PbrSurface(vec3(0.8),metal,alpha*alpha,view,nv,f0,clamp(f0.r,0.25,0.9),0.,1.5,true,vec3(0.),1.);
    var hit=Intersection();hit.n=vec3(0.,0.,1.);hit.geometricNormal=hit.n;
    var rng=jenkinsHash(id.x+971u);var sums=vec4(0.);var rejected=vec2(0.);var invalid=0.;
    for(var i=0u;i<512u;i+=1u) {
        var oldRng=rng;
        let old=samplePbrMode(surface,hit,&oldRng,false);
        let visible=samplePbrMode(surface,hit,&rng,true);
        let a=dot(old.scatter.throughput,vec3(0.2126,0.7152,0.0722));
        let b=dot(visible.scatter.throughput,vec3(0.2126,0.7152,0.0722));
        sums+=vec4(a,b,a*a,b*b);
        rejected+=vec2(select(0.,1.,a==0.),select(0.,1.,b==0.));
        let m=sampleVisibleGgx(view,alpha,rngNextFloat(&rng),rngNextFloat(&rng));
        if dot(m,view)<-1e-6 || m.z<0. || abs(dot(m,m)-1.)>1e-4 {invalid+=1.;}
    }
    queries[id.x].hit=sums/512.;
    queries[id.x].brute=vec4(rejected/512.,invalid,0.);
}
@compute @workgroup_size(64)
fn csSkyEnergy(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&queries) {return;}
    var rng=jenkinsHash(id.x+131u);var base=vec3(0.);var combined=vec3(0.);var error=0.;
    var hit=Intersection();hit.n=vec3(0.,0.,1.);hit.geometricNormal=hit.n;
    let ray=Ray(vec3(0.),vec3(0.,0.,-1.));let surface=pbrSurface(ray,hit);
    for(var i=0u;i<256u;i+=1u) {
        var oldRng=rng;
        let old=scatterPbr(ray,hit,&oldRng);
        let sampled=samplePbr(surface,hit,&rng);
        error=max(error,length(old.throughput-sampled.scatter.throughput));
        let sky=atmosphereRadiance(sampled.scatter.ray.direction);
        base+=old.throughput*sky;
        combined+=sampled.scatter.throughput*sky*misWeight(sampled.pdf,skyPdf(sampled.scatter.ray.direction));
        let d=sampleSky(&rng);let ev=evaluatePbr(surface,hit,d);let pdf=skyPdf(d);
        combined+=ev.rgb*atmosphereRadiance(d)*misWeight(pdf,ev.a)/pdf;
    }
    queries[id.x].hit=vec4(base/256.,0.);
    queries[id.x].brute=vec4(combined/256.,0.);
    queries[id.x].normal.x=error;
}
@compute @workgroup_size(64)
fn csSunEnergy(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&queries) {return;}
    var rng=jenkinsHash(id.x+517u);var direct=vec3(0.);var combined=vec3(0.);
    var hit=Intersection();hit.n=vec3(0.,1.,0.);hit.geometricNormal=hit.n;
    let ray=Ray(vec3(0.),normalize(vec3(0.3,-1.,0.1)));let surface=pbrSurface(ray,hit);
    for(var i=0u;i<256u;i+=1u) {
        // Cone samples are inside the cone by construction: constant pdf and radiance.
        let d=sampleSun(&rng);let ev=evaluatePbr(surface,hit,d);let pdf=1./skyState.sunRadiance.w;
        direct+=ev.rgb*skyState.sunRadiance.rgb/pdf;
        combined+=ev.rgb*skyState.sunRadiance.rgb*misWeight(pdf,ev.a)/pdf;
        let sampled=samplePbr(surface,hit,&rng);let b=sampled.scatter.ray.direction;
        combined+=sampled.scatter.throughput*sunRadiance(b)*misWeight(sampled.pdf,sunPdf(b));
    }
    queries[id.x].hit=vec4(direct/256.,0.);
    queries[id.x].brute=vec4(combined/256.,0.);
}
@compute @workgroup_size(64)
fn csLampEnergy(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&queries) {return;}
    var rng=jenkinsHash(id.x+733u);var nee=vec3(0.);var bsdf=vec3(0.);var combined=vec3(0.);
    var hit=Intersection();hit.n=vec3(0.,1.,0.);hit.geometricNormal=hit.n;
    let ray=Ray(vec3(0.3,1.,0.1),normalize(vec3(-0.3,-1.,-0.1)));let surface=pbrSurface(ray,hit);
    for(var i=0u;i<256u;i+=1u) {
        let s=sampleLamp(hit.p,&rng);let ev=evaluatePbr(surface,hit,s.direction);
        nee+=ev.rgb*s.radiance/s.pdf;
        // Resampled lamp choice, MIS-weighted with the source pdf.
        let r=lampRis(hit.p,hit.n,&rng);let er=evaluatePbr(surface,hit,r.direction);
        combined+=er.rgb*r.radiance*misWeight(r.pdf,er.a)*r.scale;
        let b=samplePbr(surface,hit,&rng);var h=Intersection();
        if meshIntersection(Ray(hit.p+hit.n*1e-4,b.scatter.ray.direction),&h,MAX_T) {
            let e=meshVec3(meshMaterialOffset(h.materialIdx)+4u);
            let d=normalize(b.scatter.ray.direction);
            let pdf=lampPdfAt(h.materialIdx,h.t*length(b.scatter.ray.direction),abs(dot(h.geometricNormal,d)));
            bsdf+=b.scatter.throughput*e;
            combined+=b.scatter.throughput*e*misWeight(b.pdf,pdf);
        }
    }
    queries[id.x].hit=vec4(nee/256.,0.);
    queries[id.x].brute=vec4(bsdf/256.,0.);
    queries[id.x].normal=vec4(combined/256.,0.);
}
@compute @workgroup_size(64)
fn csSky(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&queries) {return;}
    var rng=jenkinsHash(id.x+71u);var sum=vec3(0.);
    for(var i=0u;i<256u;i+=1u) {
        let d=sampleSky(&rng);
        sum+=vec3(1./skyPdf(d),d.y,dot(d,d));
    }
    queries[id.x].hit=vec4(sum/256.,0.);
}
@compute @workgroup_size(64)
fn csRoulette(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&queries) {return;}
    var rng=jenkinsHash(id.x+42u);
    var sum=vec3(0.);
    for (var i=0u;i<65536u;i+=1u) {sum+=rouletteThroughput(queries[id.x].origin.xyz,&rng);}
    queries[id.x].hit=vec4(sum/65536.,0.);
}
@compute @workgroup_size(64)
fn csCompare(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&queries) {return;}
    let ray=Ray(queries[id.x].origin.xyz,queries[id.x].direction.xyz);
    var hit=Intersection();var brute=Intersection();var t=MAX_T;
    let found=meshIntersection(ray,&hit,MAX_T);
    var bruteFound=false;
    for(var i=0u;i<meshData[1u];i+=1u) {
        var candidate=Intersection();
        if meshTriangle(ray,i,t,&candidate) {brute=candidate;t=candidate.t;bruteFound=true;}
    }
    queries[id.x].hit=vec4(select(MAX_T,hit.t,found),hit.u,hit.v,f32(hit.materialIdx));
    queries[id.x].brute=vec4(select(MAX_T,brute.t,bruteFound),brute.u,brute.v,f32(brute.materialIdx));
    queries[id.x].direction.w=f32(meshOccluded(ray,MAX_T));
    queries[id.x].normal=vec4(hit.n,f32(found));
    queries[id.x].bruteNormal=vec4(brute.n,f32(bruteFound));
}
@compute @workgroup_size(64)
fn csTextures(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&queries) {return;}
    let uv0=queries[id.x].origin.xy;let uv1=queries[id.x].origin.zw;
    // Base colour (sRGB texture) and metallic/roughness (linear copy of the same image).
    let slot=meshMaterialOffset(0u)+16u;
    queries[id.x].hit=meshTexture(slot,uv0,uv1,true);
    queries[id.x].brute=meshTexture(slot+8u,uv0,uv1,false);
}
@compute @workgroup_size(64)
fn csLight(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&queries) {return;}
    var hit=Intersection();hit.p=queries[id.x].origin.xyz;
    var rng=id.x*19u+1u;
    var direction=vec3(0.);
    if LIGHT_COUNT>0u {direction=sampleLight(hit,&rng);}
    queries[id.x].hit=vec4(direction,pdfLight(hit,queries[id.x].direction.xyz));
    queries[id.x].brute=vec4(pdfLight(hit,direction),0.,0.,0.);
}
";

fn run_queries(mesh: Mesh, queries: &[Query], entry: &str) -> Vec<Query> {
    run_scene_queries(
        Scene {
            mesh,
            ..Default::default()
        },
        queries,
        entry,
    )
}
fn run_scene_queries(scene: Scene, queries: &[Query], entry: &str) -> Vec<Query> {
    run_scene_queries_with(scene, queries, entry, &params())
}

fn run_scene_queries_with(
    scene: Scene,
    queries: &[Query],
    entry: &str,
    render_params: &crate::raytracer::RenderParams,
) -> Vec<Query> {
    let (device, queue) = gpu();
    let tracer = Raytracer::new(
        &device,
        &queue,
        &config(render_params.viewport_size),
        &scene,
        render_params,
    )
    .unwrap();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("BVH differential test"),
        source: wgpu::ShaderSource::Wgsl(
            format!(
                "{}\n{}\n{QUERY_SHADER}",
                include_str!("../raytracer.wgsl"),
                include_str!("../mesh.wgsl")
            )
            .into(),
        ),
    });
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test rays"),
        contents: bytemuck::cast_slice(queries),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 1,
            resource: buffer.as_entire_binding(),
        }],
    });
    let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[],
    });
    let sky_layout = tracer.trace_pipelines[0].get_bind_group_layout(2);
    let scene_layout = tracer.trace_pipelines[0].get_bind_group_layout(3);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[
            Some(&layout),
            Some(&empty),
            Some(&sky_layout),
            Some(&scene_layout),
        ],
        immediate_size: 0,
    });
    let light_count = scene
        .spheres
        .iter()
        .filter(|s| {
            matches!(
                scene.materials[s.material_idx as usize],
                Material::Emissive { .. }
            )
        })
        .count() as f64;
    let constants = [
        (
            "SOFTWARE_OPACITY",
            f64::from(crate::raytracer::mesh::micro_opacity()),
        ),
        ("LIGHT_COUNT", light_count),
        ("WIDE_BVH", f64::from(crate::raytracer::mesh::bvh_width())),
        (
            "QUANTIZED_BVH",
            f64::from(crate::raytracer::mesh::quantized_bvh()),
        ),
        (
            "GGX_BOUNDED",
            f64::from(std::env::var("RT_BOUNDED_GGX").is_ok()),
        ),
    ];
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &constants,
            ..Default::default()
        },
        cache: None,
    });
    let empty_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &empty,
        entries: &[],
    });
    let size = std::mem::size_of_val(queries) as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Query buffer is copied for readback; ordinary scene buffers need no COPY_SRC.
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.set_bind_group(1, &empty_group, &[]);
        pass.set_bind_group(2, &tracer.parameter_bind_group, &[]);
        pass.set_bind_group(3, &tracer.scene_bind_group, &[]);
        pass.dispatch_workgroups((queries.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buffer, 0, &readback, 0, size);
    queue.submit([encoder.finish()]);
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let result = {
        let data = readback.slice(..).get_mapped_range().unwrap();
        bytemuck::cast_slice::<u8, Query>(&data).to_vec()
    };
    readback.unmap();
    result
}

fn compare(mesh: Mesh, queries: &[Query]) -> Vec<Query> {
    let result = run_queries(mesh, queries, "csCompare");
    for (i, q) in result.iter().enumerate() {
        assert_eq!(
            q.direction[3], q.normal[3],
            "shadow visibility differs from closest hit"
        );
        assert_eq!(
            q.normal[3], q.brute_normal[3],
            "hit/miss for ray {i}: {q:?}"
        );
        for (a, b) in q
            .hit
            .iter()
            .zip(q.brute)
            .chain(q.normal.iter().zip(q.brute_normal))
        {
            assert!(
                (a - b).abs() < 1e-4,
                "BVH/brute mismatch for ray {i}: {q:?}"
            );
        }
    }
    result
}
fn ray(origin: [f32; 3], direction: [f32; 3]) -> Query {
    Query {
        origin: [origin[0], origin[1], origin[2], 0.],
        direction: [direction[0], direction[1], direction[2], 0.],
        hit: [0.; 4],
        brute: [0.; 4],
        normal: [0.; 4],
        brute_normal: [0.; 4],
    }
}

#[test]
#[ignore = "requires a GPU"]
fn gpu_bvh_matches_brute_force_including_parallel_and_inside_rays() {
    let mut rays = Vec::new();
    for y in 0..24 {
        for x in 0..32 {
            rays.push(ray(
                [x as f32 * 0.9 - 14., y as f32 * 0.8 - 9., 0.],
                [0., 0., -1.],
            ));
        }
    }
    rays.extend([
        ray([0., 0., -3.4], [0., 0., 1.]),
        ray([-13., -8., -3.], [1., 0., 0.]),
        ray([0., 0., -3.], [0., 1., 0.]),
        ray([30., 0., 0.], [0., 0., -1.]),
    ]);
    let result = compare(grid(), &rays);
    assert!(result.iter().any(|q| q.normal[3] == 1.));
    assert!(result.iter().any(|q| q.normal[3] == 0.));
    // An independent analytic triangle check at the center of the simple fixture.
    let result = compare(
        fixture(),
        &[
            ray([0., 0., 0.], [0., 0., -1.]),
            ray([20., 0., 0.], [0., 0., -1.]),
        ],
    );
    assert!((result[0].hit[0] - 3.).abs() < 1e-6);
    assert!((result[0].hit[1] - 0.5).abs() < 1e-6);
    assert!((result[0].hit[2] - 0.5).abs() < 1e-6);
    assert_eq!(result[1].normal[3], 0.);
}

#[test]
#[ignore = "requires a GPU and python scripts/download_sponza.py"]
fn sponza_bvh_matches_brute_force() {
    let mesh = Mesh::load(Path::new("assets/sponza/sponza-png.glb")).unwrap();
    let mut rays = Vec::new();
    for y in 0..16 {
        for x in 0..32 {
            let direction = glm::normalize(&glm::vec3(1., y as f32 / 8. - 0.8, x as f32 / 8. - 2.));
            rays.push(ray([-8., 1.6, 0.], direction.into()));
        }
    }
    let result = compare(mesh, &rays);
    assert!(result.iter().filter(|q| q.normal[3] > 0.).count() > 400);
}

#[test]
#[ignore = "requires a GPU"]
fn imported_emissive_triangle_renders_with_zero_sphere_lights() {
    let (device, queue) = gpu();
    let mut mesh = fixture();
    mesh.materials[0][4..7].fill(1f32.to_bits());
    let scene = Scene {
        mesh,
        ..Default::default()
    };
    let mut params = params();
    params.camera.vfov = Angle::degrees(20.);
    params.sampling.num_bounces = 1;
    let mut tracer = Raytracer::new(
        &device,
        &queue,
        &config(params.viewport_size),
        &scene,
        &params,
    )
    .unwrap();
    let pixels = render(&device, &queue, &mut tracer);
    let center = (params.viewport_size.0 as usize * (params.viewport_size.1 as usize / 2)
        + params.viewport_size.0 as usize / 2)
        * 4;
    assert!(pixels[center] > 40, "emissive mesh must be visible");
    for _ in 0..7 {
        render(&device, &queue, &mut tracer);
    }
    assert_eq!(tracer.progress(), 1.);
    params.exposure = 2.;
    tracer.set_render_params(&device, &queue, &params).unwrap();
    assert_eq!(tracer.progress(), 1., "exposure must preserve samples");
    let brighter = render(&device, &queue, &mut tracer);
    assert!(brighter[center] > pixels[center] + 40);
}

#[test]
#[ignore = "requires a GPU"]
fn gltf_alpha_mask_and_top_left_uvs_reveal_geometry_behind_cutouts() {
    let mesh = Mesh::load(Path::new("tests/assets/masked.gltf")).unwrap();
    let result = compare(
        mesh,
        &[
            ray([-0.5, 0.5, 0.], [0., 0., -1.]),
            ray([0.5, 0.5, 0.], [0., 0., -1.]),
            ray([-0.5, -0.5, 0.], [0., 0., -1.]),
            ray([0.5, -0.5, 0.], [0., 0., -1.]),
            ray([0.5, 0.5, -3.], [0., 0., 1.]),
        ],
    );
    for (q, expected) in result.iter().zip([4., 2., 2., 4., 1.]) {
        assert!((q.hit[0] - expected).abs() < 1e-6, "{q:?}");
    }
    assert!(
        result[4].normal[2] < 0.,
        "double-sided back faces reverse their normals"
    );
}

#[test]
fn gltf_reports_unsupported_required_extensions_and_blended_materials() {
    let source = std::fs::read_to_string("tests/assets/triangles.gltf").unwrap();
    let path = std::env::temp_dir().join(format!("raytracer-invalid-{}.gltf", std::process::id()));
    for (source,expected) in [
        (source.replacen('{',"{\"extensionsUsed\":[\"KHR_draco_mesh_compression\"],\"extensionsRequired\":[\"KHR_draco_mesh_compression\"],",1),"KHR_draco_mesh_compression"),
        (source.replace("\"doubleSided\": true","\"alphaMode\": \"BLEND\", \"doubleSided\": true"),"alphaMode BLEND"),
    ] {
        std::fs::write(&path,source).unwrap();
        let message=Mesh::load(&path).err().expect("must reject unsupported glTF").to_string();
        assert!(message.contains(expected),"{message}");
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
#[ignore = "requires a GPU"]
fn gltf_texture_filtering_decodes_srgb_before_interpolation_and_wraps() {
    // The metallic/roughness slot reuses the base-colour image, so the same
    // pixels are uploaded once as an sRGB texture and once as a linear one.
    fn load() -> Mesh {
        let mut mesh = Mesh::load(Path::new("tests/assets/masked.gltf")).unwrap();
        let material = &mut mesh.materials[0];
        let base: [u32; 8] = material[16..24].try_into().unwrap();
        material[24..32].copy_from_slice(&base);
        mesh
    }
    let mut mesh = load();
    mesh.pixels[1] = u32::from_le_bytes([0, 128, 0, 255]);
    mesh.textures[0][3] = 16; // repeat, nearest
    let result = run_queries(
        mesh,
        &[
            ray([0.75, 0.25, 0.], [0., 0., 0.]),
            ray([1.75, -0.75, 0.], [0., 0., 0.]),
        ],
        "csTextures",
    );
    for q in result {
        // Hardware sRGB decoding may differ from the exact curve by ~1e-3
        // (Vulkan permits approximate conversion); linear data stays exact.
        assert!((q.hit[1] - 0.215_860_5).abs() < 1e-3);
        assert!((q.brute[1] - 128. / 255.).abs() < 1e-6);
    }
    let mut mesh = load();
    mesh.pixels[1] = u32::from_le_bytes([0, 128, 0, 255]);
    mesh.textures[0][3] = 0;
    let result = run_queries(mesh, &[ray([0.5, 0.5, 0.], [0., 0., 0.])], "csTextures");
    for (a, b) in result[0]
        .hit
        .iter()
        .zip([0.5, (1. + 0.215_860_5) / 4., 0.5, 0.5])
    {
        assert!((a - b).abs() < 1e-3);
    }
    let mut mesh = load();
    mesh.textures[0][3] = 16 | 2 | (2 << 2); // mirrored repeat, nearest
    let result = run_queries(
        mesh,
        &[
            ray([1.25, 0.25, 0.], [0., 0., 0.]),
            ray([1.75, 0.25, 0.], [0., 0., 0.]),
        ],
        "csTextures",
    );
    assert!(result[0].hit[1] > 0.99);
    assert!(result[1].hit[0] > 0.99);
}

#[test]
#[ignore = "requires a GPU"]
fn sphere_light_sampling_matches_analytic_solid_angle_and_handles_zero_one_and_last_light() {
    let scene = |centers: &[glm::Vec3]| Scene {
        spheres: centers.iter().map(|&p| Sphere::new(p, 1., 0)).collect(),
        materials: vec![Material::Emissive {
            emit: Texture::new_from_color(glm::vec3(1., 1., 1.)),
        }],
        ..Default::default()
    };
    let rays = vec![ray([0., 0., 0.], [0., 0., -1.]); 128];
    let empty = run_scene_queries(scene(&[]), &rays[..1], "csLight");
    assert_eq!(empty[0].hit[3], 0.);
    let one = run_scene_queries(scene(&[glm::vec3(0., 0., -10.)]), &rays, "csLight");
    let cos_max = (1f64 - 1. / 100.).sqrt();
    let expected = (1. / (2. * std::f64::consts::PI * (1. - cos_max))) as f32;
    for q in &one {
        assert!((q.hit[3] - expected).abs() < 1e-3, "{q:?}");
        assert!(
            (q.brute[0] - expected).abs() < 1e-3,
            "sample and PDF must agree: {q:?}"
        );
        assert!(-q.hit[2] >= cos_max as f32 - 1e-6);
    }
    let two = run_scene_queries(
        scene(&[glm::vec3(0., 0., -10.), glm::vec3(10., 0., 0.)]),
        &rays,
        "csLight",
    );
    assert!(
        two.iter().any(|q| q.hit[0] > 0.99),
        "final light must be sampled"
    );
    assert!(
        two.iter().any(|q| q.hit[2] < -0.99),
        "first light must be sampled"
    );
    for q in &two {
        assert!((q.hit[3] - expected * 0.5).abs() < 1e-3);
        assert!((q.brute[0] - expected * 0.5).abs() < 1e-3);
    }
    let inside = run_scene_queries(scene(&[glm::vec3(0., 0., 0.)]), &rays[..1], "csLight");
    assert!((inside[0].hit[3] - 1. / (4. * std::f32::consts::PI)).abs() < 1e-6);
}

#[test]
#[ignore = "requires a GPU"]
fn roulette_preserves_expected_throughput_including_probability_clamps() {
    let inputs = [
        [0., 0., 0.],
        [0.01, 0.002, 0.],
        [0.4, 0.2, 0.1],
        [3., 1., 0.5],
    ];
    let queries: Vec<_> = inputs.iter().map(|&v| ray(v, [0.; 3])).collect();
    let results = run_queries(fixture(), &queries, "csRoulette");
    for (result, input) in results.iter().zip(inputs) {
        for (actual, expected) in result.hit[..3].iter().zip(input) {
            assert!(
                (actual - expected).abs() < 0.001 + expected * 0.025,
                "roulette expectation {actual} differs from {expected}"
            );
        }
    }
}

#[test]
#[ignore = "requires a GPU"]
fn sky_sampler_matches_pdf_and_integrates_constant_environment() {
    let queries = vec![ray([0.; 3], [0.; 3]); 1024];
    let result = run_queries(fixture(), &queries, "csSky");
    let mean = |channel: usize| {
        result
            .iter()
            .map(|q| f64::from(q.hit[channel]))
            .sum::<f64>()
            / 1024.
    };
    assert!((mean(0) - 4. * std::f64::consts::PI).abs() < 0.3);
    assert!((mean(2) - 1.).abs() < 1e-5);
    let sky = SkyParams::default().to_sky_state().unwrap();
    let expected_y: f64 = sky
        .sampling
        .iter()
        .enumerate()
        .map(|(i, c)| {
            f64::from(c[1])
                * (4. * std::f64::consts::PI / 2048.)
                * (1. - 2. * ((i / 64) as f64 + 0.5) / 32.)
        })
        .sum();
    assert!((mean(1) - expected_y).abs() < 0.01);
}

#[test]
#[ignore = "requires a GPU"]
fn sky_mis_preserves_unoccluded_pbr_energy_and_continuation_weights() {
    let queries = vec![ray([0.; 3], [0.; 3]); 1024];
    let results = run_queries(fixture(), &queries, "csSkyEnergy");
    for channel in 0..3 {
        let baseline: f64 = results.iter().map(|q| f64::from(q.hit[channel])).sum();
        let combined: f64 = results.iter().map(|q| f64::from(q.brute[channel])).sum();
        assert!(
            (combined / baseline - 1.).abs() < 0.025,
            "MIS changed expected sky energy: {combined} vs {baseline}"
        );
    }
    assert!(results.iter().all(|q| q.normal[0] < 1e-4));
}

#[test]
#[ignore = "requires a GPU"]
fn sun_nee_and_bsdf_hits_sum_to_the_direct_sun_integral() {
    let mut render_params = params();
    render_params.sky = crate::raytracer::SkyParams {
        zenith_degrees: 30.,
        ..crate::raytracer::SkyParams::default()
    };
    let queries = vec![ray([0.; 3], [0.; 3]); 1024];
    let scene = Scene {
        mesh: fixture(),
        ..Default::default()
    };
    let results = run_scene_queries_with(scene, &queries, "csSunEnergy", &render_params);
    for channel in 0..3 {
        let direct: f64 = results.iter().map(|q| f64::from(q.hit[channel])).sum();
        let combined: f64 = results.iter().map(|q| f64::from(q.brute[channel])).sum();
        assert!(direct > 0., "no direct sun");
        assert!(
            (combined / direct - 1.).abs() < 0.03,
            "sun MIS changed direct energy: {combined} vs {direct}"
        );
    }
}

/// Two emissive quads of different strength and size above the origin, facing down.
fn lamps() -> Mesh {
    let mut mesh = fixture();
    let template = mesh.vertices[0];
    let mut material = mesh.materials[0];
    mesh.vertices.clear();
    mesh.triangles.clear();
    for (index, (x0, size, emission)) in [(-1.2_f32, 0.6_f32, 2.0_f32), (0.5, 1.0, 0.5)]
        .into_iter()
        .enumerate()
    {
        material[4..7]
            .copy_from_slice(&[emission, emission * 0.8, emission * 0.6].map(f32::to_bits));
        mesh.materials.push(material);
        let id = mesh.materials.len() as u32 - 1;
        let offset = mesh.vertices.len() as u32;
        for (dx, dz) in [(0., 0.), (size, 0.), (0., size), (size, size)] {
            let mut v = template;
            v.position = [x0 + dx, 2. + index as f32 * 0.3, -0.4 + dz];
            v.normal = [0., -1., 0.];
            mesh.vertices.push(v);
        }
        mesh.triangles.push(Triangle {
            vertices: [offset, offset + 1, offset + 2],
            material: id,
        });
        mesh.triangles.push(Triangle {
            vertices: [offset + 1, offset + 3, offset + 2],
            material: id,
        });
    }
    mesh.build_bvh();
    mesh
}

#[test]
#[ignore = "requires a GPU"]
fn lamp_sampling_matches_bsdf_hits_and_mis_preserves_energy() {
    let queries = vec![ray([0.; 3], [0.; 3]); 2048];
    let results = run_queries(lamps(), &queries, "csLampEnergy");
    for channel in 0..3 {
        let sum = |f: &dyn Fn(&Query) -> f32| results.iter().map(|q| f64::from(f(q))).sum::<f64>();
        let nee = sum(&|q| q.hit[channel]);
        let bsdf = sum(&|q| q.brute[channel]);
        let combined = sum(&|q| q.normal[channel]);
        assert!(nee > 0., "no lamp light");
        assert!(
            (bsdf / nee - 1.).abs() < 0.05,
            "lamp sampling {nee} vs BSDF hits {bsdf}"
        );
        assert!(
            (combined / nee - 1.).abs() < 0.03,
            "resampled lamp MIS {combined} vs {nee}"
        );
    }
}

#[test]
#[ignore = "requires a GPU"]
fn vndf_preserves_energy_and_reduces_grazing_rejections() {
    let mut queries = Vec::new();
    for nv in [1.0, 0.2, 0.05] {
        for rough in [0.2, 0.5, 1.0] {
            for metal in [0.0, 1.0] {
                queries.extend(std::iter::repeat_n(ray([nv, rough, metal], [0.; 3]), 256));
            }
        }
    }
    let results = run_queries(fixture(), &queries, "csVndf");
    for group in results.chunks_exact(256) {
        let mean =
            |channel: usize| group.iter().map(|q| f64::from(q.hit[channel])).sum::<f64>() / 256.;
        let reject = |channel: usize| {
            group
                .iter()
                .map(|q| f64::from(q.brute[channel]))
                .sum::<f64>()
                / 256.
        };
        let [nv, rough, metal, _] = group[0].origin;
        assert!(
            group
                .iter()
                .all(|q| q.hit.iter().all(|v| v.is_finite()) && q.brute[2] == 0.)
        );
        assert!(
            (mean(1) / mean(0) - 1.).abs() < 0.04,
            "energy changed at nv={nv} rough={rough} metal={metal}: {} vs {}",
            mean(0),
            mean(1)
        );
        if nv < 0.1 && rough >= 0.5 {
            assert!(reject(1) < reject(0));
        }
        eprintln!(
            "VNDF nv={nv} rough={rough} metal={metal}: energy {:.6}/{:.6}, variance {:.6}/{:.6}, rejected {:.5}/{:.5}",
            mean(0),
            mean(1),
            mean(2) - mean(0) * mean(0),
            mean(3) - mean(1) * mean(1),
            reject(0),
            reject(1)
        );
    }
}
