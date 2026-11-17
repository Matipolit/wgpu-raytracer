// Wavefront path tracing ("rays never shade"): traversal and shading are
// separate dispatches over a compacted list of live paths.
//
// Per frame: csWfGenerate -> csWfArgs(0) -> for each bounce b:
//   csWfTrace(b) -> csWfShade(b) -> csWfArgs(b + 1)
// -> csWfAccumulate. The trace kernel carries no material code, so it compiles
// small; shading runs once per hit; paths that end (sky, roulette) leave the list.
//
// Random numbers are consumed in the same order as the monolithic kernel, so
// both produce the same image for the same seed.

struct WfRay { origin: vec3<f32>, pixel: u32, direction: vec3<f32>, pad: u32 }
struct WfPath { throughput: vec3<f32>, rng: u32, radiance: vec3<f32>, pad: u32 }
struct WfStage { bounce: u32, pad0: u32, pad1: u32, pad2: u32 }

@group(4) @binding(0) var<storage, read_write> wfRays: array<WfRay>;
@group(4) @binding(1) var<storage, read_write> wfPaths: array<WfPath>;
// (triangle index, t, u, v); triangle 0xffffffff is a miss.
@group(4) @binding(2) var<storage, read_write> wfHits: array<vec4<u32>>;
// Two path lists, ping-ponged by bounce parity, each one pixel count long.
@group(4) @binding(3) var<storage, read_write> wfQueues: array<u32>;
// [b]: live paths entering bounce b.
@group(4) @binding(4) var<storage, read_write> wfCounts: array<atomic<u32>>;
// Indirect dispatch arguments per bounce, 3 words each.
@group(4) @binding(5) var<storage, read_write> wfArgs: array<u32>;
@group(4) @binding(6) var<uniform> wfStage: WfStage;

const WF_GROUP = 64u;

fn wfPixelCount() -> u32 { return frameData.x * frameData.y; }
fn wfQueueBase(bounce: u32) -> u32 { return (bounce & 1u) * wfPixelCount(); }

@compute @workgroup_size(TRACE_WIDTH, TRACE_HEIGHT)
fn csWfGenerate(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frameData.x || id.y >= frameData.y { return; }
    let idx = frameData.x * id.y + id.x;
    var rng = initRng(id.xy, frameData.xy, frameData.z);
    let jitter = vec2(rngNextFloat(&rng), rngNextFloat(&rng));
    let u = (f32(id.x) + jitter.x) / f32(frameData.x);
    let v = (f32(id.y) + jitter.y) / f32(frameData.y);
    let ray = cameraMakeRay(camera, &rng, u, 1f - v);
    wfRays[idx] = WfRay(ray.origin, idx, ray.direction, 0u);
    wfPaths[idx] = WfPath(vec3(1f), rng, vec3(0f), 0u);
    // Bounce 0 visits every pixel in scanline order.
    wfQueues[idx] = idx;
}

// One invocation: dispatch arguments for the list entering wfStage.bounce, and
// a cleared counter for the list after it.
@compute @workgroup_size(1)
fn csWfArgs() {
    let b = wfStage.bounce;
    var count = atomicLoad(&wfCounts[b]);
    if b == 0u {
        count = wfPixelCount();
        atomicStore(&wfCounts[0], count);
    }
    atomicStore(&wfCounts[b + 1u], 0u);
    wfArgs[b * 3u] = (count + WF_GROUP - 1u) / WF_GROUP;
    wfArgs[b * 3u + 1u] = 1u;
    wfArgs[b * 3u + 2u] = 1u;
}

@compute @workgroup_size(WF_GROUP)
fn csWfTrace(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    quadLane = lane;
    let b = wfStage.bounce;
    if id.x >= atomicLoad(&wfCounts[b]) { return; }
    let path = wfQueues[wfQueueBase(b) + id.x];
    let r = wfRays[path];
    var best = MeshCandidate();
    var hit = vec4(0xffffffffu, 0u, 0u, 0u);
    if meshQuadCandidate(Ray(r.origin, r.direction), MAX_T, false, &best) {
        hit = vec4(best.index, bitcast<u32>(best.t), bitcast<u32>(best.u), bitcast<u32>(best.v));
    }
    wfHits[path] = hit;
}

var<workgroup> wfAlive: array<u32, WF_GROUP>;
var<workgroup> wfGroupBase: u32;

@compute @workgroup_size(WF_GROUP)
fn csWfShade(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let b = wfStage.bounce;
    let inRange = id.x < atomicLoad(&wfCounts[b]);
    var path = 0u;
    var alive = false;
    if inRange {
        path = wfQueues[wfQueueBase(b) + id.x];
        alive = wfShadePath(path, b);
    }
    // Order-preserving append: prefix within the workgroup, one global atomic.
    wfAlive[lane] = u32(alive);
    workgroupBarrier();
    if lane == 0u {
        var total = 0u;
        for (var i = 0u; i < WF_GROUP; i += 1u) { total += wfAlive[i]; }
        wfGroupBase = atomicAdd(&wfCounts[b + 1u], total);
    }
    workgroupBarrier();
    if alive {
        var prefix = 0u;
        for (var i = 0u; i < lane; i += 1u) { prefix += wfAlive[i]; }
        wfQueues[wfQueueBase(b + 1u) + wfGroupBase + prefix] = path;
    }
}

// One path vertex: emission, termination tests and BSDF sampling, mirroring
// rayColor. Returns whether the path continues to bounce b + 1.
fn wfShadePath(path: u32, b: u32) -> bool {
    var state = wfPaths[path];
    var rng = state.rng;
    let r = wfRays[path];
    var ray = Ray(r.origin, r.direction);
    var throughput = state.throughput;
    var alive = true;
    pathBounce = b;
    {
        let h = wfHits[path];
        if h.x == 0xffffffffu {
            state.radiance += throughput * skyRadiance(normalize(ray.direction));
            alive = false;
        } else {
            let hit = meshResolve(ray, MeshCandidate(bitcast<f32>(h.y), bitcast<f32>(h.z), bitcast<f32>(h.w), h.x));
            let m = meshMaterialOffset(hit.materialIdx);
            state.radiance += throughput * meshVec3(m + 4u) * meshTexture(m + 32u, vec2(hit.u, hit.v), hit.uv1, true).rgb;
            if b + 1u >= samplingParams.numBounces {
                // The final vertex's scatter would never be traced.
                alive = false;
            } else {
                let scatter = scatterPbr(ray, hit, &rng);
                throughput *= scatter.throughput;
                ray = scatter.ray;
                alive = max(throughput.r, max(throughput.g, throughput.b)) > 0.;
                // Roulette for the next vertex happens here, before its ray is
                // traced; no random numbers are drawn in between, so the
                // sequence matches the monolithic kernel.
                if alive && RUSSIAN_ROULETTE && b + 1u >= 3u {
                    throughput = rouletteThroughput(throughput, &rng);
                    alive = max(throughput.r, max(throughput.g, throughput.b)) > 0.;
                }
            }
        }
    }
    state.throughput = throughput;
    state.rng = rng;
    wfPaths[path] = state;
    if alive { wfRays[path] = WfRay(ray.origin, r.pixel, ray.direction, 0u); }
    return alive;
}

@compute @workgroup_size(TRACE_WIDTH, TRACE_HEIGHT)
fn csWfAccumulate(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frameData.x || id.y >= frameData.y { return; }
    let idx = frameData.x * id.y + id.x;
    var pixel = wfPaths[idx].radiance;
    if samplingParams.clearAccumulatedSamples == 0u {
        pixel += vec3(imageBuffer[idx][0], imageBuffer[idx][1], imageBuffer[idx][2]);
    }
    imageBuffer[idx] = array<f32, 3>(pixel.r, pixel.g, pixel.b);
}
