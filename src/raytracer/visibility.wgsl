// Rasterized primary visibility (`--visibility`, see visibility.rs).
// A raster pass writes one triangle index per pixel (a visibility buffer); a compute
// pass intersects each pixel's camera ray with that triangle and with the analytic
// spheres, then fills the G-buffer through the tracer's own meshResolve and
// writeGbufferHit, so rasterized and traced first hits share every shading rule.
@group(6) @binding(0) var<storage, read> visList: array<u32>;
@group(6) @binding(1) var visTarget: texture_2d<u32>;
// Sub-pixel jitter of this frame's raster (0.5 = pixel centres) and the frame size.
struct VisFrame { jitter: vec2<f32>, size: vec2<f32> }
@group(6) @binding(2) var<uniform> visFrame: VisFrame;

// Indexed draws over the mesh's own vertices (shared vertices are shaded once); a
// draw's first instance is its offset in visList, and primitive_index the triangle
// within the draw.
struct VisVertex {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) base: u32,
    @location(1) world: vec3<f32>,
}

// Clip position for the tracer's pinhole camera. w is the ray parameter t at which
// the primary ray reaches p (1 on the image plane), so depth = MIN_T / t gives a
// reverse-Z buffer whose near plane is exactly the tracer's minimum hit distance.
fn visClip(p: vec3<f32>) -> vec4<f32> {
    let w = camera.lowerLeftCorner - camera.eye;
    let n = cross(camera.horizontal, camera.vertical);
    let d = p - camera.eye;
    let t = dot(d, n) / dot(w, n);
    let s = (dot(d, camera.horizontal) - t * dot(w, camera.horizontal)) / dot(camera.horizontal, camera.horizontal);
    let v = (dot(d, camera.vertical) - t * dot(w, camera.vertical)) / dot(camera.vertical, camera.vertical);
    // Shift the image so each pixel centre samples the jittered position.
    let shift = 2. * vec2(0.5 - visFrame.jitter.x, visFrame.jitter.y - 0.5) / visFrame.size;
    return vec4(2. * s - t + shift.x * t, 2. * v - t + shift.y * t, MIN_T, t);
}

@vertex
fn vsVisibility(@location(0) p: vec3<f32>, @builtin(instance_index) base: u32) -> VisVertex {
    var out: VisVertex;
    out.position = visClip(p);
    out.base = base;
    out.world = p;
    return out;
}

// Opaque triangles: hardware culling (single-sided) or none (double-sided).
@fragment
fn fsVisibility(in: VisVertex, @builtin(primitive_index) primitive: u32) -> @location(0) u32 {
    return visList[in.base + primitive];
}

// Alpha-tested triangles: the tracer's back-face rule and alpha test (meshCandidate).
@fragment
fn fsVisibilityTested(in: VisVertex, @builtin(primitive_index) primitive: u32) -> @location(0) u32 {
    let triangle = visList[in.base + primitive];
    let accel = meshData[8u] + triangle * 12u;
    let flags = meshData[accel + 11u];
    let tri = meshData[2u] + triangle * 4u;
    let va = vertexOffset(meshData[tri]);
    let vb = vertexOffset(meshData[tri + 1u]);
    let vc = vertexOffset(meshData[tri + 2u]);
    let a = meshVec3(va);
    let e1 = meshVec3(vb) - a;
    let e2 = meshVec3(vc) - a;
    if (flags & 1u) == 0u && dot(in.world - camera.eye, cross(e1, e2)) >= 0. { discard; }
    // Barycentrics of the interpolated surface point.
    let w = in.world - a;
    let d11 = dot(e1, e1); let d12 = dot(e1, e2); let d22 = dot(e2, e2);
    let w1 = dot(w, e1); let w2 = dot(w, e2);
    let inv = 1. / (d11 * d22 - d12 * d12);
    let bu = (d22 * w1 - d12 * w2) * inv;
    let bv = (d11 * w2 - d12 * w1) * inv;
    let material = meshMaterialOffset(meshData[accel + 7u]);
    let weights = vec3(1. - bu - bv, bu, bv);
    let uv0 = weights.x * vertexUV0(va) + weights.y * vertexUV0(vb) + weights.z * vertexUV0(vc);
    let uv1 = weights.x * vertexUV1(va) + weights.y * vertexUV1(vb) + weights.z * vertexUV1(vc);
    let alpha = meshFloat(material + 3u) * meshTexture(material + 16u, uv0, uv1, false).a;
    if alpha < meshFloat(material + 10u) { discard; }
    return triangle;
}

@compute @workgroup_size(8, 8)
fn csVisibilityResolve(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frameData.x || id.y >= frameData.y { return; }
    gbufferPixel = frameData.x * id.y + id.x;
    let u = (f32(id.x) + visFrame.jitter.x) / f32(frameData.x);
    let v = (f32(id.y) + visFrame.jitter.y) / f32(frameData.y);
    gbufferSample = vec2(u * f32(frameData.x), v * f32(frameData.y));
    var rng = 1u;
    let ray = cameraMakeRay(camera, &rng, u, 1. - v);
    // Mip level from the pixel footprint (ray cone of the primary ray): filtered
    // instead of aliased albedo, and far less texture traffic in large scenes.
    if RAY_CONES {
        coneActive = true;
        coneWidth = 0.;
        coneSpread = length(camera.vertical) / (f32(frameData.y) * max(length(camera.lowerLeftCorner + 0.5 * camera.horizontal + 0.5 * camera.vertical - camera.eye), 1e-5));
    }
    var hit = Intersection();
    if visibilityHit(ray, id.xy, &hit) { writeGbufferHit(ray, hit); } else { writeGbufferMiss(ray); }
}

fn visibilityJitter() -> vec2<f32> { return visFrame.jitter; }

// Primary-ray mesh candidate (VIS_PRIMARY): the rasterized triangle for this pixel,
// intersected exactly with the ray (Moller-Trumbore); barycentrics are clamped because
// raster coverage rules decide edge pixels.
fn visibilityCandidate(ray: Ray, tmax: f32, best: ptr<function, MeshCandidate>) -> bool {
    if VIS_OUTPUT {
        // The raster is at output resolution: the texel under this sample's position.
        let texel = gbufferSample * visFrame.size / vec2<f32>(frameData.xy);
        return visibilityCandidateAt(ray, min(vec2<u32>(max(texel, vec2(0.))), vec2<u32>(visFrame.size) - vec2(1u)), tmax, best);
    }
    return visibilityCandidateAt(ray, vec2(gbufferPixel % frameData.x, gbufferPixel / frameData.x), tmax, best);
}

fn visibilityCandidateAt(ray: Ray, pixel: vec2<u32>, tmax: f32, best: ptr<function, MeshCandidate>) -> bool {
    let triangle = textureLoad(visTarget, pixel, 0).r;
    if triangle == 0xffffffffu { return false; }
    let tri = meshData[2u] + triangle * 4u;
    let a = meshVec3(vertexOffset(meshData[tri]));
    let e1 = meshVec3(vertexOffset(meshData[tri + 1u])) - a;
    let e2 = meshVec3(vertexOffset(meshData[tri + 2u])) - a;
    let p = cross(ray.direction, e2);
    let inv = 1. / dot(e1, p);
    let delta = ray.origin - a;
    let q = cross(delta, e1);
    let t = dot(e2, q) * inv;
    let bu = clamp(dot(delta, p) * inv, 0., 1.);
    let bv = clamp(dot(ray.direction, q) * inv, 0., 1. - bu);
    if !(t > MIN_T && t < tmax) { return false; }
    *best = MeshCandidate(t, bu, bv, triangle, 0u);
    return true;
}

// G-buffer-only resolve: the rasterized hit, then spheres.
fn visibilityHit(ray: Ray, pixel: vec2<u32>, outHit: ptr<function, Intersection>) -> bool {
    var hit = Intersection();
    var closest = MAX_T;
    var found = false;
    var candidate = MeshCandidate();
    var meshFound = TRACE_MESHES && visibilityCandidateAt(ray, pixel, MAX_T, &candidate);
    // Moving objects are not rasterized: trace them in front of the rasterized hit.
    meshFound = meshInstanceCandidate(ray, select(MAX_T, candidate.t, meshFound), false, &candidate) || meshFound;
    if meshFound {
        hit = meshResolve(ray, candidate);
        closest = candidate.t;
        found = true;
    }
    for (var s = 0u; s < SPHERE_COUNT; s += 1u) {
        var sphere = Intersection();
        if rayIntersectSphere(ray, s, MIN_T, closest, &sphere) {
            hit = sphere;
            closest = sphere.t;
            found = true;
        }
    }
    *outHit = hit;
    return found;
}

// GPU occlusion culling of clusters (opt-in, RT_VIS_OCCLUSION=1; two passes, see visibility.rs):
// pass 1 draws clusters visible last frame; a coarse depth pyramid (farthest depth per
// 16x16 tile) of that result tests every other cluster in the frustum, and pass 2
// draws the ones that may be visible. The second test also refreshes the visible set.
struct VisCluster {
    lo: vec3<f32>,
    firstIndex: u32,
    hi: vec3<f32>,
    indexCount: u32,
    triangleOffset: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}
@group(6) @binding(8) var<storage, read> visClusters: array<VisCluster>;
@group(6) @binding(9) var<storage, read_write> visArgs1: array<u32>;
@group(6) @binding(10) var<storage, read_write> visArgs2: array<u32>;
@group(6) @binding(11) var<storage, read_write> visVisible: array<u32>;
@group(6) @binding(12) var visDepth: texture_depth_2d;
@group(6) @binding(13) var<storage, read_write> visHiz: array<f32>;
const VIS_TILE = 16u;

// Frustum test in clip space: clip coordinates are affine in world space, so a box is
// outside when all eight corners are beyond one plane (x = +-w, y = +-w, w = MIN_T).
fn visClusterClip(c: VisCluster) -> array<vec4<f32>, 8> {
    var corners: array<vec4<f32>, 8>;
    for (var k = 0u; k < 8u; k += 1u) {
        let p = select(c.lo, c.hi, vec3((k & 1u) != 0u, (k & 2u) != 0u, (k & 4u) != 0u));
        corners[k] = visClip(p);
    }
    return corners;
}

fn visInFrustum(corners: array<vec4<f32>, 8>) -> bool {
    var outside = vec4(true);
    var behind = true;
    for (var k = 0u; k < 8u; k += 1u) {
        let q = corners[k];
        outside = outside & vec4((q.x < -q.w), (q.x > q.w), (q.y < -q.w), (q.y > q.w));
        behind = behind && q.w < MIN_T;
    }
    return !any(outside) && !behind;
}

// DrawIndexedIndirectArgs: index count, instances, first index, base vertex, first instance.
fn visArgs(c: VisCluster, draw: bool) -> array<u32, 5> {
    return array<u32, 5>(select(0u, c.indexCount, draw), 1u, c.firstIndex, 0u, c.triangleOffset);
}

@compute @workgroup_size(64)
fn csVisCull1(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&visClusters) { return; }
    let c = visClusters[i];
    let args = visArgs(c, visVisible[i] != 0u && visInFrustum(visClusterClip(c)));
    for (var k = 0u; k < 5u; k += 1u) { visArgs1[i * 5u + k] = args[k]; }
}

var<workgroup> visTileMin: atomic<u32>;

// Farthest (smallest, reverse-Z) depth of each 16x16 tile after pass 1.
@compute @workgroup_size(16, 16)
fn csVisHiz(@builtin(workgroup_id) tile: vec3<u32>, @builtin(local_invocation_index) lane: u32, @builtin(local_invocation_id) local: vec3<u32>) {
    if lane == 0u { atomicStore(&visTileMin, 0x7f800000u); }
    workgroupBarrier();
    let size = vec2<u32>(visFrame.size);
    let p = tile.xy * VIS_TILE + local.xy;
    if p.x < size.x && p.y < size.y {
        atomicMin(&visTileMin, bitcast<u32>(textureLoad(visDepth, p, 0)));
    }
    workgroupBarrier();
    if lane == 0u {
        let tiles = (size.x + VIS_TILE - 1u) / VIS_TILE;
        visHiz[tile.y * tiles + tile.x] = bitcast<f32>(atomicLoad(&visTileMin));
    }
}

// Whether any part of the box may be in front of the pass-1 depth.
fn visMayBeVisible(corners: array<vec4<f32>, 8>) -> bool {
    var lo = vec2(1e30);
    var hi = vec2(-1e30);
    var nearest = 0.;
    for (var k = 0u; k < 8u; k += 1u) {
        let q = corners[k];
        if q.w <= MIN_T { return true; }
        let ndc = q.xy / q.w;
        let pixel = vec2(0.5 + 0.5 * ndc.x, 0.5 - 0.5 * ndc.y) * visFrame.size;
        lo = min(lo, pixel);
        hi = max(hi, pixel);
        nearest = max(nearest, MIN_T / q.w);
    }
    let size = vec2<u32>(visFrame.size);
    let tiles = (size + vec2(VIS_TILE - 1u)) / VIS_TILE;
    let t0 = vec2<u32>(clamp(floor(lo / f32(VIS_TILE)), vec2(0.), vec2<f32>(tiles - vec2(1u))));
    let t1 = vec2<u32>(clamp(floor(hi / f32(VIS_TILE)), vec2(0.), vec2<f32>(tiles - vec2(1u))));
    if (t1.x - t0.x + 1u) * (t1.y - t0.y + 1u) > 64u { return true; }
    for (var y = t0.y; y <= t1.y; y += 1u) {
        for (var x = t0.x; x <= t1.x; x += 1u) {
            if nearest >= visHiz[y * tiles.x + x] { return true; }
        }
    }
    return false;
}

@compute @workgroup_size(64)
fn csVisCull2(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&visClusters) { return; }
    let c = visClusters[i];
    let corners = visClusterClip(c);
    let visible = visInFrustum(corners) && visMayBeVisible(corners);
    visVisible[i] = select(0u, 1u, visible);
    let args = visArgs(c, visible && visArgs1[i * 5u] == 0u);
    for (var k = 0u; k < 5u; k += 1u) { visArgs2[i * 5u + k] = args[k]; }
}
