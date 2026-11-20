// Mesh textures are a binding array (see mesh.wgsl and gpu_textures.rs).
enable wgpu_binding_array;

const EPSILON = 0.001f;

const PI = 3.1415927f;
const FRAC_1_PI = 0.31830987f;
const FRAC_PI_2 = 1.5707964f;

const MIN_T = 0.001f;
const MAX_T = 1000f;

const CHANNEL_R = 0u;
const CHANNEL_G = 1u;
const CHANNEL_B = 2u;

@group(0) @binding(0) var<uniform> vertexUniforms: VertexUniforms;

@vertex
fn vsMain(model: VertexInput) -> VertexOutput {
    return VertexOutput(
        vertexUniforms.viewProjectionMat * vertexUniforms.modelMat * vec4<f32>(model.position, 0.0, 1.0),
        model.texCoords
    );
}

struct VertexUniforms {
    viewProjectionMat: mat4x4<f32>,
    modelMat: mat4x4<f32>,
}

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) texCoords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clipPosition: vec4<f32>,
    @location(0) texCoords: vec2<f32>,
}

@group(1) @binding(0) var<uniform> frameData: vec4<u32>;
@group(1) @binding(1) var<storage, read_write> imageBuffer: array<array<f32, 3>>;
@group(1) @binding(2) var<storage, read> accumulatedImage: array<array<f32, 3>>;
@group(1) @binding(3) var<storage, read_write> filteredImage: array<array<f32, 3>>;
@group(1) @binding(4) var<storage, read> displayImage: array<array<f32, 3>>;
// [0]: log-average luminance of the filtered image (written by csExposure),
// [1]: nonzero when automatic exposure is enabled (written by the host).
@group(1) @binding(5) var<storage, read_write> exposureState: array<f32, 4>;
@group(1) @binding(6) var<storage, read> displayExposure: array<f32, 4>;

@group(2) @binding(0) var<uniform> camera: Camera;
@group(2) @binding(1) var<uniform> samplingParams: SamplingParams;
@group(2) @binding(2) var<storage, read> skyState: SkyState;
// The camera of the previous frame, for motion vectors.
@group(2) @binding(3) var<uniform> prevCamera: Camera;

@group(3) @binding(0) var<storage, read> spheres: array<Sphere>;
@group(3) @binding(1) var<storage, read> materials: array<Material>;
@group(3) @binding(2) var<storage, read> textures: array<array<f32, 3>>;
@group(3) @binding(3) var<storage, read> lights: array<u32>;
// Sphere centres in the previous frame (xyz), for motion vectors of moving spheres.
@group(3) @binding(7) var<storage, read> spherePrevCenters: array<vec4<f32>>;
// Primary-hit guides for the temporal denoiser, three vec4 per pixel:
// (albedo.rgb, distance), (normal.xyz, roughness), (motion.xy in pixels, metallic, hit kind).
// Distance 0 and kind 0 mark a sky pixel; kind 1 a mesh, 2 a sphere.
@group(1) @binding(7) var<storage, read_write> gbuffer: array<vec4<f32>>;

// Specialized by the host; dispatch dimensions use the same constants.
override SPHERE_COUNT: u32 = 0u;
override LIGHT_COUNT: u32 = 0u;
override TRACE_MESHES: bool = false;
override TRACE_SPHERES: bool = true;
override TRACE_WIDTH: u32 = 8u;
override TRACE_HEIGHT: u32 = 8u;
// Opt-in unbiased path termination, retaining the existing hard bounce limit.
override RUSSIAN_ROULETTE: bool = true;
override SKY_IMPORTANCE: bool = false;
// Mesh surfaces sample the sun disk directly (shadow ray), MIS with BSDF hits.
override SUN: bool = false;
// Mesh surfaces sample emissive scene triangles (the lamp table) directly.
override LAMPS: bool = false;
// Input for a temporal neural denoiser: the frame's mean (not sum) of samples, with
// directly visible emission removed and stored after the guides (gbuffer[3N + i]);
// csTdEmission adds it back after denoising, so emitters are never filtered.
override DENOISER_INPUT: bool = false;
// Captures (`--capture-emission`): directly visible emission is also stored after the
// guides, the radiance keeps it (the capture subtracts it from the 1-spp input).
override CAPTURE_EMISSION: bool = false;
// ReSTIR DI for lamps at the primary surface (compiled when the scene has lamps,
// active when samplingParams.restir != 0): per-pixel reservoirs persist across
// frames (temporal reuse along motion) and borrow from previous-frame neighbours
// (spatial reuse), so each pixel effectively weighs hundreds of lamp candidates.
override RESTIR: bool = false;
override RESTIR_SPATIAL: u32 = 6u;
override RESTIR_RADIUS: f32 = 16.;
override RESTIR_CAP: f32 = 20.;
// Reservoirs: three sets (final of the last two frames by parity, then this frame's
// initial ones) x pixels x 3 vec2<u32>: (triangle, barycentrics unorm16x2),
// (W f32, M f32), (distance from the camera f32, normal oct snorm16x2).
@group(1) @binding(8) var<storage, read_write> reservoirs: array<vec2<u32>>;
var<private> sampleIndex: u32;
// ReSTIR GI (real-time frames, mesh scenes): the path's light arriving at the
// primary surface through its first bounce is recorded as a sample (secondary point,
// its outgoing radiance, the bounce pdf) instead of being added to the pixel;
// csRestirGi reuses samples across neighbours and frames and adds the result.
override RESTIR_GI: bool = false;
// GI_BLOCKS: one pixel per 2x2 block (rotating each frame) traces the bounce and
// records a sample; the other three end their path at the primary surface and get
// their indirect light from reuse alone (records with only the primary point).
override GI_BLOCKS: bool = false;
override GI_SPATIAL: u32 = 4u;
// History length (M) below which the neighbours are consulted (RT_GI_SPATIAL_BELOW;
// 1e9: always, as before).
override GI_SPATIAL_BELOW: f32 = 8.;
const GI_RADIUS = 24.;
// History length cap (M), in samples (RT_GI_CAP), while the view moves. While it is
// still the cap grows with the samples traced since it stopped (giCap), so a pixel's
// estimate keeps improving instead of forgetting all but its last 20 samples: the
// still-view average (temporal.rs) converges instead of keeping its grain.
override GI_CAP: f32 = 20.;
override GI_STILL_GROWTH: bool = true;
// Pixels per frame above which a pixel counts as moving (RT_GI_MOVING, 0: never).
override GI_MOVING: f32 = 0.25;
// Shading reuse (RT_GI_SHADE neighbours, 0: off). A pixel's reservoir keeps a sample for
// about 4 x GI_CAP frames, and a bright one for seconds (its weight decays by 20/21 per
// new sample), so the 1-spp frames repeat their error: in the living room a pixel's
// noise is 88% the same one frame later and 62% the same 16 frames later, and a
// 64-frame average looks like one frame. The denoiser turns those standing bright
// pixels into blobs that crawl as the reservoirs change, and the still-view average
// (temporal.rs) keeps them as speckles. With this, what a pixel shows is resampled
// each frame from the reservoirs of GI_SHADE random pixels within GI_SHADE_RADIUS on
// the same surface (last frame's, around where the pixel was; same tangent plane and
// a normal within GI_SHADE_NORMAL), not from its own, with a visibility ray. Its own
// sample is shown only when no neighbour qualifies or the neighbours' pick is hidden.
// The stored history is not touched (fed back, the pooled weights grow without
// bound). Living room / day Bistro, still camera: error left in the mean of 16 frames
// per pixel -62% / -48%, frame-to-frame correlation 0.88 / 0.49 -> 0.05 / 0.03;
// brightness of a long average -2.6% / -1.5% (light a pixel sees and its neighbours
// do not is under-counted; without the normal test -5%).
override GI_SHADE: u32 = 3u;
override GI_SHADE_RADIUS: f32 = 24.;
override GI_SHADE_OWN_LIMIT: f32 = 3.;
override GI_SHADE_NORMAL: f32 = 0.9;
// GI_VISIBILITY_REUSE: a borrowed winner that is occluded is stored with W = 0 (its M
// still counts, diluting the next frame). Off: stored with its weight, visibility
// applied to the shading only (no measurable difference: Sponza, living room).
// GI_TEMPORAL_TRUST: the pixel's own reprojected history (tested when it was chosen)
// gets its visibility ray only every 4th frame (staggered over 2x2 pixels), so a
// moving occluder leaks its light for at most 4 frames. Steam Deck, Medium: GI pass
// night Bistro 16.9 -> 8.6 ms (every frame skipped), Sponza 8.1 -> 5.8 ms.
override GI_VISIBILITY_REUSE: bool = true;
override GI_TEMPORAL_TRUST: bool = true;
// A bounce sample that carries nothing (it left the hemisphere, or was never traced)
// counts as the pixel's sample with weight 0, and the pixel still reuses its history
// and its neighbours. Off (RT_GI_NULL=0): such a pixel gets no bounce light that frame.
override GI_NULL_SAMPLES: bool = true;
// What cannot be reused here is dropped instead of counted as a sample that found no
// light: a neighbour's sample outside the Jacobian's range is skipped, and a pick that
// turns out hidden is forgotten (not kept with weight 0 and its full history length).
// Counted, they dilute the pixel's good samples: day Bistro was 12% too dark overall
// and up to 50% on facades and foliage; 1% with this. Off: RT_GI_DROP=0.
override GI_DROP_UNUSABLE: bool = true;
// Three vec4 per pixel of this frame's samples (primary point + normal, secondary
// point + normal, radiance rg/b + pdf), then two sets (parity) of final reservoirs,
// two vec4 per pixel (secondary point + normal, radiance rg/b|M, W, distance).
@group(1) @binding(9) var<storage, read_write> giData: array<vec4<u32>>;
// The resolve at the GI sample's secondary point keeps shading normals even when it
// is the last bounce (SKIP_FINAL_SCATTER), so light can be sampled there.
var<private> pathKeepLast: bool;
// Shadow (visibility) queries pass through transmissive surfaces.
var<private> shadowRay: bool;
// The path is inside a refracting volume (KHR_materials_volume), absorbing pathSigma
// per unit length.
var<private> pathInside: bool;
var<private> pathSigma: vec3<f32>;
var<private> primaryEmission: vec3<f32>;
// Share of light samples given to the sun when lamps are sampled too.
const SUN_SHARE = 0.75;
// Lamp candidates per light sample (resampled importance sampling, the first stage
// of ReSTIR DI): one shadow ray goes to the candidate chosen by unshadowed
// contribution, so nearby, facing lamps win over distant bright ones.
override LAMP_CANDIDATES: u32 = 4u;
// Candidates at deeper bounces (their noise matters less).
override LAMP_CANDIDATES_DEEP: u32 = 2u;
override GGX_VNDF: bool = false;
override GGX_BOUNDED:bool=false;

override SPARSE_DENOISE:bool=false;
// Diagnostics: output (BVH nodes visited, triangles tested, rays traced) per pixel instead of radiance.
override TRAVERSAL_STATS:bool=false;
// Write primary-hit guides and motion vectors (temporal denoiser input).
override GBUFFER:bool=false;
// Reference frames (reference.rs): after a dispatch the G-buffer's albedo and normal
// are the means over its samples (pixel-integrated guides), not the last sample's.
override REFERENCE_GUIDES:bool=false;
override COHERENCE_TEST:bool=false;
var<private> gbufferPixel:u32;
var<private> gbufferSample:vec2<f32>;
// Set at a bounce-0 mesh hit: the G-buffer takes the material values the mesh shading
// already sampled (shaded*, set by scatterPbr/pbrSurface) instead of sampling twice.
var<private> gbufferPending:bool;
var<private> shadedBase:vec3<f32>;
var<private> shadedRough:f32;
var<private> shadedMetal:f32;
var<private> statNodes:u32;
var<private> statTriangles:u32;
var<private> statRays:u32;
override WIDE_BVH:u32=2u;
override PAIR_BVH:bool=false;
override QUAD_BVH:bool=false;
// Compressed 8-wide BVH (--cwbvh; mesh.wgsl meshCwCandidate).
override CW_BVH:bool=false;
override OCT_BVH:bool=false;
// Hardware ray queries (--hw-rt); see hwrt.wgsl.
override HW_RT:bool=false;
// With HW_RT: which queries use the hardware structure (the others traverse the
// compute BVH, which is built too): closest hits, shadow (any-hit) rays.
override HW_CLOSEST:bool=true;
override HW_SHADOWS:bool=true;
override HW_FLIP_FACING:bool=false;
override QUAD_CULL:bool=false;
override QUAD_LEAF_COUNT:bool=false;
override QUAD_Q8:bool=false;
// Workgroup-memory traversal stack (RT_QUAD_SLM): entry i of lane l at i*64+l.
var<private> quadLane: u32;
const QUAD_CULL_SLOTS=64u;
override QUANTIZED_BVH:bool=false;
override OPACITY_APPROX:bool=false;
override SOFTWARE_OPACITY:bool=false;
override ALPHA_CAP:bool=false;
var<private> alphaQueries:u32;
var<private> pathDiffuse:bool;
var<private> pathBounce:u32;
override PACKED_SHADING:bool=false;
override STOCHASTIC_TEXTURE:u32=0u;
override PACKED_DIRECTIONS:bool=false;
override SKIP_FINAL_SCATTER:bool=false;
// With SKIP_FINAL_SCATTER the path's last vertex had no normal at all, so its light
// sample evaluated to nothing: without bounce reuse (the Low preset) a path's bounce
// brought back only sky and emission, and Sponza's shade was at 4% of its brightness.
// The last vertex is now lit with its geometric normal; the bounce-reuse sample's
// secondary point uses the same cheap resolve. RT_FLAT_LAST=0: the old behaviour.
override FLAT_LAST:bool=true;
// The atmosphere lights the path's last vertex too (RT_SKY_LAST=0: as before, where
// "N bounces" gave the sun and the lamps one more path segment than the sky; found by
// comparing against Blender Cycles with equal settings).
override SKY_LAST:bool=true;
override BLUE_NOISE_JITTER:bool=false;
// `--blue-noise-path`: the first BLUE_NOISE_DIMS random numbers of every path (pixel
// jitter, lens, bounce-0 lobe and direction) come from the 64x64 blue-noise masks,
// shifted per dimension (R2 offsets) and per frame (golden ratio), so 1-spp error
// is spread as blue noise in screen space and low-discrepancy over frames.
override BLUE_NOISE_PATH:bool=false;
// `--center-jitter`: primary rays through pixel centres (rasterized-visibility comparisons).
override CENTER_JITTER:bool=false;
// `--visibility-primary`: bounce 0 takes its hit from the rasterized visibility buffer
// (visibility.wgsl) instead of tracing the primary ray; all pixels share the raster's
// per-frame sub-pixel jitter.
override VIS_PRIMARY:bool=false;
// VIS_OUTPUT (joint network): the visibility raster is at output resolution with its own
// jitter; render-resolution samples keep the shared frame jitter and read the texel under
// their position (visibility.wgsl visibilityCandidate).
override VIS_OUTPUT:bool=false;
fn primaryJitter() -> vec2<f32> {
    if VIS_OUTPUT {return frameJitter(frameData.z);}
    return visibilityJitter();
}
// Upscaling (`--render-scale`): every pixel uses this frame's sub-pixel jitter, so the
// upscaler knows where each sample lies (frameJitter, identical to Rust's frame_jitter).
override SHARED_JITTER:bool=false;
fn frameJitter(frame: u32) -> vec2<f32> {
    var result = vec2(0.);
    for (var axis = 0u; axis < 2u; axis += 1u) {
        let base = 2u + axis;
        var i = frame % 1024u + 1u;
        var f = 1.;
        var r = 0.;
        while i > 0u {
            f /= f32(base);
            r += f * f32(i % base);
            i /= base;
        }
        result[axis] = r;
    }
    return result;
}
// Set for the primary ray: the next mesh query takes its candidate from the visibility
// buffer instead of traversing (spheres and hit resolution stay shared).
var<private> visibilityPending:bool;
// VISIBILITY STUBS: removed when visibility.wgsl is part of the module.
fn visibilityCandidate(ray: Ray, tmax: f32, best: ptr<function, MeshCandidate>) -> bool { return false; }
fn visibilityJitter() -> vec2<f32> { return vec2(0.5); }
// END VISIBILITY STUBS
const BLUE_NOISE_DIMS = 8u;
var<private> blueNoiseDim:u32;
var<private> blueNoisePixel:vec2<u32>;
var<private> blueNoiseShift:f32;
override RAY_CONES:bool=false;
// Ray cones only at the primary hit (later bounces sample the base level).
override CONES_PRIMARY_ONLY:bool=false;
// Whether the current hit selects mips from its ray cone.
var<private> coneActive:bool;
var<private> coneWidth:f32;
var<private> coneSpread:f32;
var<private> footprintUV0:vec2<f32>;
var<private> footprintUV1:vec2<f32>;
var<private> textureRng:u32;
// Every invocation owns one pixel; the next pass only sees completed sums.
@compute @workgroup_size(TRACE_WIDTH, TRACE_HEIGHT)
fn csAccumulate(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32, @builtin(num_workgroups) groups: vec3<u32>) {
    quadLane = lane;
    let origin = vec2(samplingParams.traceOrigin & 0xffffu, samplingParams.traceOrigin >> 16u);
    var at = id.xy;
    let grid = groups.xy * vec2(TRACE_WIDTH, TRACE_HEIGHT);
    // The host dispatches an even grid for this (any workgroup shape).
    if GI_BLOCKS && RESTIR_GI && samplingParams.restir != 0u && all(grid % vec2(2u) == vec2(0u)) {
        // The grid as four quarter-resolution planes, one per position in the 2x2
        // blocks, the GI owners first: a wave holds owners only (they trace the
        // bounce) or none, instead of one busy lane in four.
        let half = grid / 2u;
        let side = vec2<u32>(id.xy >= half);
        let turn = (samplingParams.restir + 2u) & 3u;
        let owner = ((turn & 1u) ^ (origin.x & 1u)) + 2u * ((turn >> 1u) ^ (origin.y & 1u));
        let position = (owner + side.x + 2u * side.y) & 3u;
        at = 2u * (id.xy - side * half) + vec2(position & 1u, position >> 1u);
    }
    let x = at.x + origin.x;
    let y = at.y + origin.y;
    if x >= frameData.x || y >= frameData.y {
        return;
    }
    let idx = frameData.x * y + x;
    gbufferPixel = idx;
    var pixel = vec3(0f);
    if samplingParams.clearAccumulatedSamples == 0u {
        pixel = vec3(imageBuffer[idx][0], imageBuffer[idx][1], imageBuffer[idx][2]);
    }
    var rngState = initRng(vec2(x, y), frameData.xy, frameData.z);
    pixel += samplePixel(x, y, &rngState);
    imageBuffer[idx] = array<f32, 3>(pixel.r, pixel.g, pixel.b);
}

// Joint network (upscale.rs): this frame's guides at output resolution, bound with an
// output-sized frameData and the output guide buffer in place of `gbuffer`. Primary
// rays with the frame's shared jitter (as the full-resolution guides of the training
// captures, SHARED_JITTER); motion in output pixels.
@compute @workgroup_size(8, 8)
fn csOutputGuides(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    quadLane = lane;
    if id.x >= frameData.x || id.y >= frameData.y {return;}
    gbufferPixel = frameData.x * id.y + id.x;
    var rng = initRng(id.xy, frameData.xy, frameData.z ^ 0x68e31da4u);
    // Primary hits from the output-resolution raster when there is one (same jitter).
    var jitter = frameJitter(frameData.z);
    if VIS_PRIMARY {jitter = visibilityJitter(); visibilityPending = true;}
    gbufferSample = vec2<f32>(id.xy) + jitter;
    let u = gbufferSample.x / f32(frameData.x);
    let v = gbufferSample.y / f32(frameData.y);
    if RAY_CONES {coneActive = false;}
    let ray = cameraMakeRay(camera, &rng, u, 1. - v);
    var hit = Intersection();
    if intersection(ray, &hit) {writeGbufferHit(ray, hit);} else {writeGbufferMiss(ray);}
}

fn averagePixel(x: u32, y: u32) -> vec3<f32> {
    let pixel = accumulatedImage[frameData.x * y + x];
    let invN = 1f / f32(max(1u, samplingParams.accumulatedSamplesPerPixel));
    return invN * vec3(pixel[0], pixel[1], pixel[2]);
}

// Fallback for API callers requesting a kernel larger than the UI's radius 5.
fn filterGlobal(x: u32, y: u32) -> vec3<f32> {
    let average = averagePixel(x, y);
    let radius = i32(samplingParams.denoisingRadius);
    let sigma = max(0.0001f, samplingParams.denoisingSigmaColor);
    var colorSum = vec3(0f);
    var weightSum = 0f;
    for (var dy = -radius; dy <= radius; dy += 1) {
        for (var dx = -radius; dx <= radius; dx += 1) {
            let nx = i32(x) + dx;
            let ny = i32(y) + dy;
            if nx >= 0 && nx < i32(frameData.x) && ny >= 0 && ny < i32(frameData.y) {
                let neighbor = averagePixel(u32(nx), u32(ny));
                let difference = average - neighbor;
                let weight = exp(-dot(difference, difference) / (2f * sigma * sigma));
                colorSum += neighbor * weight;
                weightSum += weight;
            }
        }
    }
    return colorSum / max(weightSum, 0.0001f);
}

// 8x8 pixels plus a radius-5 halo. Only the requested halo is loaded.
var<workgroup> filterTile: array<vec3<f32>, 324>;

@compute @workgroup_size(8, 8)
fn csDenoise(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let inBounds = id.x < frameData.x && id.y < frameData.y;
    let radius = samplingParams.denoisingRadius;
    let enabled = samplingParams.denoisingEnabled != 0u && radius != 0u &&
        samplingParams.accumulatedSamplesPerPixel > 1u;
    if !enabled {
        if inBounds {
            let color = averagePixel(id.x, id.y);
            filteredImage[frameData.x * id.y + id.x] = array<f32, 3>(color.r, color.g, color.b);
        }
        return;
    }
    if SPARSE_DENOISE && radius==1u {
        if inBounds {let color=sparseBilateral(id.x,id.y);filteredImage[frameData.x*id.y+id.x]=array<f32,3>(color.r,color.g,color.b);}
        return;
    }
    if radius > 5u {
        if inBounds {
            let color = filterGlobal(id.x, id.y);
            filteredImage[frameData.x * id.y + id.x] = array<f32, 3>(color.r, color.g, color.b);
        }
        return;
    }

    let stride = 8u + 2u * radius;
    let origin = vec2<i32>(group.xy * 8u) - vec2(i32(radius));
    for (var index = lane; index < stride * stride; index += 64u) {
        let point = origin + vec2<i32>(i32(index % stride), i32(index / stride));
        var color = vec3(0f);
        if point.x >= 0 && point.y >= 0 && point.x < i32(frameData.x) && point.y < i32(frameData.y) {
            color = averagePixel(u32(point.x), u32(point.y));
        }
        filterTile[index] = color;
    }
    // Edge invocations must participate in loading and in the barrier too.
    workgroupBarrier();
    if !inBounds {
        return;
    }
    let center = local.xy + vec2(radius);
    let average = filterTile[center.y * stride + center.x];
    let sigma = max(0.0001f, samplingParams.denoisingSigmaColor);
    var colorSum = vec3(0f);
    var weightSum = 0f;
    for (var dy = -i32(radius); dy <= i32(radius); dy += 1) {
        for (var dx = -i32(radius); dx <= i32(radius); dx += 1) {
            let point = vec2<i32>(id.xy) + vec2(dx, dy);
            if point.x >= 0 && point.y >= 0 && point.x < i32(frameData.x) && point.y < i32(frameData.y) {
                let offset = vec2<i32>(center) + vec2(dx, dy);
                let neighbor = filterTile[u32(offset.y) * stride + u32(offset.x)];
                let difference = average - neighbor;
                let weight = exp(-dot(difference, difference) / (2f * sigma * sigma));
                colorSum += neighbor * weight;
                weightSum += weight;
            }
        }
    }
    let color = colorSum / max(weightSum, 0.0001f);
    filteredImage[frameData.x * id.y + id.x] = array<f32, 3>(color.r, color.g, color.b);
}

// Log-average ("key") luminance over a fixed 64x64 grid of 4x4 blocks of the filtered image,
// so the cost does not grow with resolution. One workgroup; runs after filtering.
var<workgroup> exposureSums: array<f32, 256>;

@compute @workgroup_size(256)
fn csExposure(@builtin(local_invocation_index) lane: u32) {
    let width = frameData.x;
    let height = frameData.y;
    let columns = min(width, 64u);
    let rows = min(height, 64u);
    var sum = 0f;
    for (var i = lane; i < columns * rows; i += 256u) {
        // Average a 4x4 block first: the log of single noisy low-spp pixels
        // (many black, few bright) would bias the key low and overexpose.
        let x0 = min((i % columns) * width / columns, max(width, 4u) - 4u);
        let y0 = min((i / columns) * height / rows, max(height, 4u) - 4u);
        var luminance = 0f;
        for (var y = y0; y < min(y0 + 4u, height); y += 1u) {
            for (var x = x0; x < min(x0 + 4u, width); x += 1u) {
                let color = filteredImage[width * y + x];
                luminance += dot(vec3(color[0], color[1], color[2]), vec3(0.2126, 0.7152, 0.0722));
            }
        }
        sum += log(max(luminance / f32((min(y0 + 4u, height) - y0) * (min(x0 + 4u, width) - x0)), 1e-4));
    }
    exposureSums[lane] = sum;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if lane < stride {
            exposureSums[lane] += exposureSums[lane + stride];
        }
        workgroupBarrier();
    }
    if lane == 0u {
        exposureState[0] = exposureSums[0] / f32(columns * rows);
    }
}

// The remaining raster pass only tone-maps the computed image and draws the UI.
@fragment
fn fsMain(in: VertexOutput) -> @location(0) vec4<f32> {
    let x = u32(in.clipPosition.x);
    let y = u32(in.clipPosition.y);
    if x >= frameData.x || y >= frameData.y {
        return vec4(0f, 0f, 0f, 1f);
    }
    let color = displayImage[frameData.x * y + x];
    var c = vec3(color[0], color[1], color[2]);
    if displaySettings.strength > 0. {
        c = mix(c, bloomAt(x, y), displaySettings.strength);
    }
    return vec4(tonemap(c), 1f);
}

// Display settings and the bloom pyramid's half-resolution sum (bloom.rs).
struct DisplaySettings {tonemapper: u32, strength: f32, levels: u32, pad: u32}
@group(2) @binding(10) var bloomTexture: texture_2d<f32>;
@group(2) @binding(11) var<uniform> displaySettings: DisplaySettings;
@group(2) @binding(12) var bloomSampler: sampler;
// It sums `levels` levels: their mean.
fn bloomAt(x: u32, y: u32) -> vec3<f32> {
    let uv = (vec2(f32(x), f32(y)) + vec2(0.5)) / vec2(f32(frameData.x), f32(frameData.y));
    return textureSampleLevel(bloomTexture, bloomSampler, uv, 0.).rgb / f32(max(displaySettings.levels, 1u));
}

// Exposure, then the selected display transform; linear output (sRGB surface).
fn tonemap(x: vec3<f32>) -> vec3<f32> {
    let automatic = select(1f, 0.18 / exp(max(displayExposure[0], log(0.005))), displayExposure[1] != 0f);
    // The default compensation (2) offsets Hable's dark curve; the other transforms
    // map middle grey to a mid display value by design, so they take half.
    let exposed = 0.5 * bitcast<f32>(frameData.w) * automatic * max(x, vec3(0.));
    switch displaySettings.tonemapper {
        case 0u: {return agx(exposed);}
        case 1u: {return acesFitted(exposed);}
        case 2u: {return pbrNeutral(exposed);}
        default: {return uncharted2(x);}
    }
}

// AgX (Troy Sobotka), the minimal fit by Benjamin Wrensch (iolite engine): inset to
// the AgX space, log2 encoding over [-12.47, 4.03] EV, a polynomial sigmoid, outset,
// then the 2.2 display EOTF back to linear.
fn agx(color: vec3<f32>) -> vec3<f32> {
    let inset = mat3x3(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                       0.0784335999999992, 0.878468636469772, 0.0784336,
                       0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    let outset = mat3x3(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                        -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                        -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    let minEv = -12.47393;let maxEv = 4.026069;
    var v = inset * color;
    v = clamp(log2(max(v, vec3(1e-10))), vec3(minEv), vec3(maxEv));
    v = (v - minEv) / (maxEv - minEv);
    let v2 = v * v;let v4 = v2 * v2;
    v = 15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2 + 0.1191 * v - 0.00232;
    v = outset * v;
    return pow(clamp(v, vec3(0.), vec3(1.)), vec3(2.2));
}

// ACES filmic: Stephen Hill's fit of the RRT + sRGB ODT.
fn acesFitted(color: vec3<f32>) -> vec3<f32> {
    let input = mat3x3(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777);
    let output = mat3x3(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602);
    let v = input * color;
    let rrt = (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.4329510) + 0.238081);
    return clamp(output * rrt, vec3(0.), vec3(1.));
}

// Khronos PBR Neutral (https://github.com/KhronosGroup/ToneMapping).
fn pbrNeutral(color: vec3<f32>) -> vec3<f32> {
    let start = 0.8 - 0.04;let desaturation = 0.15;
    let low = min(color.r, min(color.g, color.b));
    let offset = select(0.04, low - 6.25 * low * low, low < 0.08);
    var c = color - offset;
    let peak = max(c.r, max(c.g, c.b));
    if peak < start {return clamp(c, vec3(0.), vec3(1.));}
    let d = 1. - start;
    let newPeak = 1. - d * d / (peak + d - start);
    c *= newPeak / peak;
    let g = 1. - 1. / (desaturation * (peak - newPeak) + 1.);
    return clamp(mix(c, vec3(newPeak), g), vec3(0.), vec3(1.));
}

fn uncharted2(x: vec3<f32>) -> vec3<f32> {
    // Based on uncharted2 tonemapping function
    // https://dmnsgn.github.io/glsl-tone-map/
    // Automatic exposure maps the log-average luminance to middle grey (0.18);
    // the user exposure then acts as compensation on top of it. Adaptation stops at
    // a key of 0.005 (daylight scenes measure 0.1-0.7, night Bistro 0.0008), so night
    // stays dark instead of being brightened to dusk, noise included.
    let automatic = select(1f, 0.18 / exp(max(displayExposure[0], log(0.005))), displayExposure[1] != 0f);
    let exposureBias = bitcast<f32>(frameData.w) * automatic;
    let curr = uncharted2Tonemap(exposureBias * x);

    let w = 11.2;
    let whiteScale = 1f / uncharted2Tonemap(vec3(w));
    return whiteScale * curr;
}

fn uncharted2Tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 0.15;
    let b = 0.50;
    let c = 0.10;
    let d = 0.20;
    let e = 0.02;
    let f = 0.30;
    let w = 11.2;
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f;
}

fn samplePixel(x: u32, y: u32, rngState: ptr<function, u32>) -> vec3<f32> {
    let imageWidth = frameData.x;
    let imageHeight = frameData.y;
    let invWidth = 1f / f32(imageWidth);
    let invHeight = 1f / f32(imageHeight);

    let numSamples = samplingParams.numSamplesPerPixel;
    var color = vec3(0f);
    var emission = vec3(0f);
    var guideAlbedo = vec3(0f);
    var guideNormal = vec3(0f);
    for (var i = 0u; i < numSamples; i += 1u) {
        sampleIndex = i;
        if BLUE_NOISE_PATH {
            blueNoiseDim = 0u;
            blueNoisePixel = vec2(x, y);
            blueNoiseShift = fract(0.6180339887 * f32((frameData.z * numSamples + i) % 65536u));
        }
        var jitter=vec2(rngNextFloat(rngState),rngNextFloat(rngState));
        if CENTER_JITTER {jitter=vec2(0.5);}
        // One shared position per frame at 1 spp (the upscaler's input); dispatches of
        // many samples (references) keep a random position per sample, so pixels are
        // integrated over their whole area, not over one position per dispatch.
        if SHARED_JITTER && numSamples == 1u {jitter=frameJitter(frameData.z);}
        if VIS_PRIMARY {jitter=primaryJitter();}
        if BLUE_NOISE_JITTER {
            let mask=unpack2x16unorm(meshData[meshData[14u]+(y%64u)*64u+(x%64u)]);
            let sequence=fract(vec2(0.7548776662,0.569840291)*f32(frameData.z%65536u));
            jitter=fract(mask+sequence);
        }
        let u = (f32(x) + jitter.x) * invWidth;
        let v = (f32(y) + jitter.y) * invHeight;

        if STOCHASTIC_TEXTURE!=0u {textureRng=jenkinsHash(*rngState^0xa7391u);}
        if RAY_CONES {
            coneWidth=0.;coneSpread=length(camera.vertical)/(f32(imageHeight)*max(length(camera.lowerLeftCorner+0.5*camera.horizontal+0.5*camera.vertical-camera.eye),1e-5));
        }
        let primaryRay = cameraMakeRay(camera, rngState, u, 1f - v);
        gbufferSample = vec2(u * f32(imageWidth), v * f32(imageHeight));
        primaryEmission = vec3(0.);
        color += rayColor(primaryRay, rngState);
        if DENOISER_INPUT {color -= primaryEmission;emission += primaryEmission;}
        else if CAPTURE_EMISSION {emission += primaryEmission;}
        if REFERENCE_GUIDES {guideAlbedo += gbuffer[gbufferPixel * 3u].xyz;guideNormal += gbuffer[gbufferPixel * 3u + 1u].xyz;}
    }
    if REFERENCE_GUIDES && numSamples > 0u {
        let n = 1. / f32(numSamples);
        gbuffer[gbufferPixel * 3u] = vec4(guideAlbedo * n, gbuffer[gbufferPixel * 3u].w);
        gbuffer[gbufferPixel * 3u + 1u] = vec4(guideNormal * n, gbuffer[gbufferPixel * 3u + 1u].w);
    }
    if DENOISER_INPUT && numSamples > 0u {
        let n = 1. / f32(numSamples);
        gbuffer[frameData.x * frameData.y * 3u + gbufferPixel] = vec4(emission * n, 0.);
        color *= n;
    }
    if CAPTURE_EMISSION && !DENOISER_INPUT && numSamples > 0u {
        gbuffer[frameData.x * frameData.y * 3u + gbufferPixel] = vec4(emission / f32(numSamples), 0.);
    }
    if TRAVERSAL_STATS {return vec3(f32(statNodes),f32(statTriangles),f32(statRays));}
    return color;
}

// A surviving path is reweighted so the expected contribution is unchanged.
fn rouletteThroughput(throughput: vec3<f32>, rngState: ptr<function, u32>) -> vec3<f32> {
    let survival = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 0.95);
    if rngNextFloat(rngState) >= survival { return vec3(0.); }
    return throughput / survival;
}

fn rayColor(primaryRay: Ray, rngState: ptr<function, u32>) -> vec3<f32> {
    var ray = primaryRay;

    var color = vec3(0f);
    var throughput = vec3(1f);
    var skyMisWeight = 1.;
    var sunMisWeight = 1.;
    // Pdf of the BSDF sample that continued the path (< 0: no light sampling there).
    var lastBsdfPdf = -1.;
    pathInside = false;pathSigma = vec3(0.);
    let sunOn = SUN && skyState.sunRadiance.g > 0.;
    let sunShare = select(0.,select(1.,SUN_SHARE,LAMPS),sunOn);
    // Light-sample shares of the vertex the path continues from (for MIS of what the
    // continuation hits): ReSTIR vertices sample the sun every time and cover lamps.
    var vertexSunShare = sunShare;
    var lampsCovered = false;
    let giActive = RESTIR_GI && sampleIndex == 0u && samplingParams.restir != 0u;
    var giRecorded = false;var giHasSecondary = false;var giPrimaryOnly = false;var giNull = false;
    let giTurn = (samplingParams.restir + 2u) & 3u;
    let giOwner = !GI_BLOCKS || ((gbufferPixel % frameData.x) & 1u) == (giTurn & 1u) && ((gbufferPixel / frameData.x) & 1u) == (giTurn >> 1u);
    var giColor0 = vec3(0.);var giThroughput = vec3(0.);var giPdf = 0.;
    var giPrimary = vec3(0.);var giPrimaryNormal = vec3(0.);
    var giSecondary = vec3(0.);var giSecondaryNormal = vec3(0.);
    pathKeepLast = false;
    if OPACITY_APPROX {pathDiffuse=false;}
    var packedDirection=0u;var packedValid=false;

    for (var bounce = 0u; bounce < samplingParams.numBounces; bounce += 1u) {
        if SKIP_FINAL_SCATTER {pathBounce=bounce;}
        if RUSSIAN_ROULETTE && bounce >= 3u {
            throughput = rouletteThroughput(throughput, rngState);
            if max(throughput.r, max(throughput.g, throughput.b)) <= 0. { break; }
        }
        if PACKED_DIRECTIONS && packedValid {ray.direction=decodeOct(packedDirection);}
        if RAY_CONES {coneActive=!CONES_PRIMARY_ONLY || bounce==0u;}
        var hit = Intersection();
        if VIS_PRIMARY {visibilityPending = bounce == 0u;}
        pathKeepLast = giRecorded && bounce == 1u && !FLAT_LAST;

        if intersection(ray, &hit) {
            if pathInside {throughput *= exp(-pathSigma * hit.t * length(ray.direction));}
            if giRecorded && bounce == 1u {giSecondary = hit.p;giSecondaryNormal = hit.n;giHasSecondary = true;}
            if GBUFFER && bounce == 0u {
                let shaded = TRACE_MESHES && hit.sphereIdx == 0xffffffffu && !(SKIP_FINAL_SCATTER && samplingParams.numBounces == 1u);
                if shaded { gbufferPending = true; } else { writeGbufferHit(ray, hit); }
            }
            if TRACE_MESHES && hit.sphereIdx == 0xffffffffu {
                let m = meshMaterialOffset(hit.materialIdx);
                let emission = meshVec3(m+4u);
                if any(emission > vec3(0.)) {
                    var weight = 1.;
                    if lampsCovered && bounce > 0u {
                        weight = 0.;
                    } else if LAMPS && lastBsdfPdf >= 0. {
                        let d = normalize(ray.direction);
                        let lampPdf = (1.-sunShare)*lampPdfAt(hit.materialIdx,hit.t*length(ray.direction),abs(dot(hit.geometricNormal,d)));
                        weight = misWeight(lastBsdfPdf,lampPdf);
                    }
                    let emitted = throughput * emission * meshTexture(m+32u,vec2(hit.u,hit.v),hit.uv1,true).rgb * weight;
                    color += emitted;
                    if bounce == 0u {primaryEmission = emitted;}
                }
                let last = bounce+1u==samplingParams.numBounces;
                // MIS state of the vertex before this one: a delta transmission sample
                // passes straight through and keeps it (shadow rays skip transmissive
                // surfaces, so that vertex's light samples already see through glass).
                let before = array(lastBsdfPdf,skyMisWeight,sunMisWeight,vertexSunShare);
                let lampsCoveredBefore = lampsCovered;
                vertexSunShare = sunShare;lampsCovered = false;
                if RESTIR && bounce == 0u && sampleIndex == 0u && samplingParams.restir != 0u {
                    let surface = pbrSurface(ray,hit);
                    if sunOn {
                        let direction = sampleSun(rngState);
                        let eval = evaluatePbr(surface,hit,direction);
                        let pdf = 1./skyState.sunRadiance.w;
                        if any(eval.rgb > vec3(0.)) && !occludedWithin(Ray(hit.p+hit.geometricNormal*0.0001,direction),MAX_T) {
                            color += throughput * skyState.sunRadiance.rgb * eval.rgb * select(misWeight(pdf,eval.a),1.,last)/pdf;
                        }
                    }
                    color += throughput * restirLamps(hit,surface,rngState);
                    vertexSunShare = select(0.,1.,sunOn);lampsCovered = true;
                } else
                if sunOn || LAMPS {
                    // One light sample (sun or lamp) at every surface; MIS against the
                    // BSDF continuation, which is never traced from the last surface.
                    let surface = pbrSurface(ray,hit);
                    var direction = vec3(0.);var radiance = vec3(0.);var pdf = 0.;var scale = 0.;var limit = MAX_T;
                    if rngNextFloat(rngState) < sunShare {
                        direction = sampleSun(rngState);
                        radiance = skyState.sunRadiance.rgb;
                        pdf = sunShare/skyState.sunRadiance.w; // inside the cone by construction
                        scale = 1./pdf;
                    } else {
                        let lamp = lampRis(hit.p,hit.n,select(LAMP_CANDIDATES_DEEP,LAMP_CANDIDATES,bounce == 0u),rngState);
                        direction = lamp.direction;radiance = lamp.radiance;limit = lamp.distance*0.999;
                        // MIS uses the source pdf; RIS supplies the estimator's weight.
                        pdf = (1.-sunShare)*lamp.pdf;
                        scale = lamp.scale/(1.-sunShare);
                    }
                    let eval = evaluatePbr(surface,hit,direction);
                    if scale > 0. && any(eval.rgb*radiance > vec3(0.)) && !occludedWithin(Ray(hit.p+hit.geometricNormal*0.0001,direction),limit) {
                        let weight = select(misWeight(pdf,eval.a),1.,last);
                        color += throughput * radiance * eval.rgb * weight*scale;
                    }
                }
                if SKY_LAST && last {
                    // Nothing scatters on from the last vertex, so a BSDF sample that
                    // would escape never brings back the sky: the sun and the lamps lit
                    // this vertex (light samples), the atmosphere did not. One
                    // visibility ray in a BSDF-sampled direction adds it (the sun's
                    // disk is the light sample's).
                    let toSky = samplePbr(pbrSurface(ray,hit),hit,rngState).scatter;
                    let escaped = normalize(toSky.ray.direction);
                    if max(toSky.throughput.r,max(toSky.throughput.g,toSky.throughput.b)) > 0. && !skyOccluded(Ray(toSky.ray.origin,escaped)) {
                        color += throughput * toSky.throughput * atmosphereRadiance(escaped);
                    }
                    if SKIP_FINAL_SCATTER {break;}
                }
                if SKIP_FINAL_SCATTER && last {
                    // All mesh BSDFs consume exactly three path random draws.
                    for(var unused=0u;unused<3u;unused+=1u){_ = rngNextInt(rngState);}
                    break;
                }
                var scatter = Scatter();
                if SKY_IMPORTANCE {
                    let surface = pbrSurface(ray,hit);
                    if bounce + 1u < samplingParams.numBounces {
                        let direction = sampleSky(rngState);
                        let eval = evaluatePbr(surface,hit,direction);
                        let pdf = skyPdf(direction);
                        if any(eval.rgb > vec3(0.)) && !skyOccluded(Ray(hit.p+hit.geometricNormal*0.0001,direction)) {
                            color += throughput * atmosphereRadiance(direction) * eval.rgb * misWeight(pdf,eval.a)/pdf;
                        }
                    }
                    let sampled = samplePbr(surface,hit,rngState);
                    scatter = sampled.scatter;
                    skyMisWeight = misWeight(sampled.pdf,skyPdf(scatter.ray.direction));
                    if sunOn {sunMisWeight = misWeight(sampled.pdf,vertexSunShare*sunPdf(normalize(scatter.ray.direction)));}
                    lastBsdfPdf = sampled.pdf;
                } else if sunOn || LAMPS || giActive {
                    let sampled = samplePbr(pbrSurface(ray,hit),hit,rngState);
                    scatter = sampled.scatter;
                    if sunOn {sunMisWeight = misWeight(sampled.pdf,vertexSunShare*sunPdf(normalize(scatter.ray.direction)));}
                    lastBsdfPdf = sampled.pdf;
                } else {scatter = scatterPbr(ray,hit,rngState);}
                if (SKY_IMPORTANCE || sunOn || LAMPS || giActive) && lastBsdfPdf >= 1e7 {
                    lastBsdfPdf = before[0];skyMisWeight = before[1];sunMisWeight = before[2];
                    vertexSunShare = before[3];lampsCovered = lampsCoveredBefore;
                }
                if GBUFFER && gbufferPending {
                    gbufferPending = false;
                    writeGbuffer(shadedBase, hit.t * length(ray.direction), hit.n, shadedRough, shadedMetal, meshPrevious, 1.);
                }
                if giActive && bounce == 0u && lastBsdfPdf < 1e7 && meshFloat(m+48u) <= 0. && (GI_NULL_SAMPLES || lastBsdfPdf > 0.) {
                    giPrimary = hit.p;giPrimaryNormal = hit.n;
                    // Not this block's owner this frame: the bounce comes from reuse.
                    if !giOwner {giPrimaryOnly = true;break;}
                    // A sample that carries nothing (below the surface: no pdf or no
                    // throughput) is still this pixel's sample: recorded as a null one,
                    // so the pixel keeps reusing instead of going black for the frame.
                    if lastBsdfPdf > 0. && max(scatter.throughput.r,max(scatter.throughput.g,scatter.throughput.b)) > 0. {
                        giRecorded = true;giColor0 = color;giThroughput = scatter.throughput;giPdf = lastBsdfPdf;
                    } else {giNull = true;}
                }
                ray = scatter.ray;
                if RAY_CONES {coneSpread=min(1.,coneSpread+0.25*meshFloat(m+8u));}
                if PACKED_DIRECTIONS {packedDirection=pack2x16snorm(encodeOct(normalize(ray.direction)));ray.direction=vec3(0.);packedValid=true;}
                throughput *= scatter.throughput;
                if max(throughput.r,max(throughput.g,throughput.b)) <= 0. {break;}
                continue;
            }
            if TRACE_SPHERES {
                let material = materials[hit.materialIdx];

                if material.id == 4u {
                    let emissionTexture = material.desc1;
                    let emissionColor = textureLookup(emissionTexture, hit.u, hit.v);
                    color += throughput * emissionColor;
                    if bounce == 0u {primaryEmission = throughput * emissionColor;}
                    break;
                }

                skyMisWeight = 1.;sunMisWeight = 1.;lastBsdfPdf = -1.;lampsCovered = false;packedValid=false;if OPACITY_APPROX {pathDiffuse=false;}
                var scatter = scatterRay(ray, hit, material, rngState);
                ray = scatter.ray;
                throughput *= scatter.throughput;
                if max(throughput.r,max(throughput.g,throughput.b)) <= 0. {break;}
            }
        } else {
            if GBUFFER && bounce == 0u { writeGbufferMiss(ray); }
            let escaped = normalize(ray.direction);
            color += throughput * (atmosphereRadiance(escaped) * skyMisWeight + sunRadiance(escaped) * sunMisWeight);
            // An escaped GI bounce: a distant secondary point in that direction.
            if giRecorded && bounce == 1u {giSecondary = giPrimary + escaped * 1e5;giSecondaryNormal = -escaped;giHasSecondary = true;}

            break;
        }
    }

    if giActive {
        var record = array(vec4(0u), vec4(0u), vec4(0u));
        if giRecorded && giHasSecondary {
            // Radiance leaving the secondary point toward the primary one.
            let outgoing = select(vec3(0.), (color - giColor0) / giThroughput, giThroughput > vec3(1e-8));
            record = array(
                vec4(bitcast<vec3<u32>>(giPrimary), pack2x16snorm(encodeOct(giPrimaryNormal))),
                vec4(bitcast<vec3<u32>>(giSecondary), pack2x16snorm(encodeOct(giSecondaryNormal))),
                vec4(pack2x16float(min(max(outgoing.rg, vec2(0.)), vec2(60000.))), pack2x16float(vec2(min(max(outgoing.b, 0.), 60000.), 0.)), bitcast<u32>(giPdf), 1u),
            );
            color = giColor0;
        } else if giPrimaryOnly || (GI_NULL_SAMPLES && (giNull || giRecorded)) {
            record[0] = vec4(bitcast<vec3<u32>>(giPrimary), pack2x16snorm(encodeOct(giPrimaryNormal)));
            record[2] = vec4(0u, 0u, 0u, select(3u, 2u, giPrimaryOnly));
        }
        for (var k = 0u; k < 3u; k += 1u) {giData[gbufferPixel * 3u + k] = record[k];}
    }
    return color;
}

fn intersection(ray: Ray, intersection: ptr<function, Intersection>) -> bool {
    if ALPHA_CAP {alphaQueries=0u;}
    var closestT = MAX_T;
    var closestIntersection = Intersection();

    for (var idx = 0u; idx < SPHERE_COUNT; idx = idx + 1u) {
        var testIntersect = Intersection();
        if rayIntersectSphere(ray, idx, MIN_T, closestT, &testIntersect) {
            closestT = testIntersect.t;
            closestIntersection = testIntersect;
        }
    }

    var meshHit=Intersection();
    if TRACE_MESHES && meshIntersection(ray,&meshHit,closestT) {
        closestIntersection=meshHit;
        closestT=meshHit.t;
    }
    if closestT < MAX_T {
        *intersection = closestIntersection;
        return true;
    }

    return false;
}

fn scatterRay(wo: Ray, hit: Intersection, material: Material, rngState: ptr<function, u32>) -> Scatter {
    switch material.id {
        case 0u: {
            let texture = material.desc1;
            return scatterMixtureDensity(hit, texture, rngState);
        }

        case 1u: {
            let texture = material.desc1;
            let fuzz = material.x;
            return scatterMetal(wo, hit, texture, fuzz, rngState);
        }

        case 2u: {
            let refractionIndex = material.x;
            return scatterDielectric(wo, hit, refractionIndex, rngState);
        }

        case 3u: {
            let texture1 = material.desc1;
            let texture2 = material.desc2;
            return scatterCheckerboard(hit, texture1, texture2, rngState);
        }

        default: {
            return scatterMissingMaterial(hit, rngState);
        }
    }
}

fn scatterMixtureDensity(hit: Intersection, albedo: TextureDescriptor, rngState: ptr<function, u32>) -> Scatter {
    if LIGHT_COUNT==0u {
        return Scatter(Ray(hit.p,sampleLambertian(hit,rngState)),textureLookup(albedo,hit.u,hit.v));
    }
    let scatterDirection = sampleMixtureDensity(hit, rngState);
    let materialValue = evalLambertian(hit, albedo, scatterDirection);
    let materialPdf = pdfLambertian(hit, scatterDirection);
    let lightPdf = pdfLight(hit, scatterDirection);
    let throughput = materialValue / max(EPSILON, (0.5f * materialPdf + 0.5f * lightPdf));
    return Scatter(Ray(hit.p, scatterDirection), throughput);
}

fn sampleMixtureDensity(hit: Intersection, rngState: ptr<function, u32>) -> vec3<f32> {
    if rngNextFloat(rngState) < 0.5f {
        return sampleLambertian(hit, rngState);
    } else {
        return sampleLight(hit, rngState);
    }
}

fn evalLambertian(hit: Intersection, texture: TextureDescriptor, wi: vec3<f32>) -> vec3<f32> {
    return textureLookup(texture, hit.u, hit.v) * FRAC_1_PI * max(0., dot(hit.n, wi));
}

fn sampleLambertian(hit: Intersection, rngState: ptr<function, u32>) -> vec3<f32> {
    let v = rngNextInCosineWeightedHemisphere(rngState);
    let onb = pixarOnb(hit.n);
    return onb * v;
}

fn pdfLambertian(hit: Intersection, wi: vec3<f32>) -> f32 {
    return max(0., dot(hit.n, wi) * FRAC_1_PI);
}

fn sampleLight(hit: Intersection, rngState: ptr<function, u32>) -> vec3<f32> {
    // Select a random light using a uniform distribution.
    let numLights = LIGHT_COUNT;
    let lightIdx = rngNextUintInRange(rngState, 0u, numLights);
    let sphereIdx = lights[lightIdx];
    let sphere = spheres[sphereIdx];

    return sampleHemisphere(hit, sphere, rngState);
}

// Uniform solid-angle sampling of a sphere, including the inside case.
fn sampleHemisphere(hit: Intersection, sphere: Sphere, rngState: ptr<function, u32>) -> vec3<f32> {
    let toCenter = sphere.centerAndPad.xyz - hit.p;
    let distanceSqr = dot(toCenter,toCenter);
    let radiusSqr = sphere.radius*sphere.radius;
    let r1 = rngNextFloat(rngState);
    let r2 = rngNextFloat(rngState);
    if distanceSqr <= radiusSqr {
        let z = 1.-2.*r1;
        let radial = sqrt(max(0.,1.-z*z));
        return vec3(radial*cos(2.*PI*r2),radial*sin(2.*PI*r2),z);
    }
    let cosMax = sqrt(max(0.,1.-radiusSqr/distanceSqr));
    // Stable form of 1-cosMax for small/distant lights.
    let coneWidth = radiusSqr / (distanceSqr*(1.+cosMax));
    let z = 1.-r1*coneWidth;
    let radial = sqrt(max(0.,1.-z*z));
    return pixarOnb(normalize(toCenter))*vec3(radial*cos(2.*PI*r2),radial*sin(2.*PI*r2),z);
}

fn pdfLight(hit: Intersection, wi: vec3<f32>) -> f32 {
    var pdf = 0.;
    let numLights = LIGHT_COUNT;
    if numLights==0u { return 0.; }
    // Sum the mixture of actual light PDFs, without an unnecessary scene ray.
    for(var i=0u;i<numLights;i+=1u) {
        let sphere=spheres[lights[i]];
        let toCenter=sphere.centerAndPad.xyz-hit.p;
        let distanceSqr=dot(toCenter,toCenter);
        let radiusSqr=sphere.radius*sphere.radius;
        if distanceSqr<=radiusSqr { pdf+=1./(4.*PI); }
        else {
            let cosMax=sqrt(max(0.,1.-radiusSqr/distanceSqr));
            if dot(normalize(toCenter),wi)>=cosMax {
                let coneWidth=radiusSqr/(distanceSqr*(1.+cosMax));
                pdf+=1./(2.*PI*coneWidth);
            }
        }
    }
    return pdf/f32(numLights);
}

fn pixarOnb(n: vec3<f32>) -> mat3x3<f32> {
    // https://www.jcgt.org/published/0006/01/01/paper-lowres.pdf
    let s = select(-1f, 1f, n.z >= 0f);
    let a = -1f / (s + n.z);
    let b = n.x * n.y * a;
    let u = vec3<f32>(1f + s * n.x * n.x * a, s * b, -s * n.x);
    let v = vec3<f32>(b, s + n.y * n.y * a, -n.y);

    return mat3x3<f32>(u, v, n);
}

fn scatterMetal(wo: Ray, hit: Intersection, texture: TextureDescriptor, fuzz: f32, rngState: ptr<function, u32>) -> Scatter {
    let scatterDirection = reflect(wo.direction, hit.n) + fuzz * rngNextVec3InUnitSphere(rngState);
    let albedo = textureLookup(texture, hit.u, hit.v);
    return Scatter(Ray(hit.p, scatterDirection), albedo);
}

fn scatterDielectric(rayIn: Ray, hit: Intersection, refractionIndex: f32, rngState: ptr<function, u32>) -> Scatter {
    let wo = rayIn.direction;
    var outwardNormal = vec3(0f);
    var niOverNt = 0f;
    var cosine = 0f;
    if dot(wo, hit.n) > 0f {
        outwardNormal = -hit.n;
        niOverNt = refractionIndex;
        cosine = refractionIndex * dot(normalize(wo), hit.n);
    } else {
        outwardNormal = hit.n;
        niOverNt = 1f / refractionIndex;
        cosine = dot(normalize(-wo), hit.n);
    };

    var refractedDirection = vec3(0f);
    if refract(wo, outwardNormal, niOverNt, &refractedDirection) {
        let reflectionProb = schlick(cosine, refractionIndex);
        var wi = refractedDirection;
        if rngNextFloat(rngState) < reflectionProb {
            wi = reflect(wo, hit.n);
        }

        return Scatter(Ray(hit.p, wi), vec3(1f));
    }

    let wi = reflect(wo, hit.n);
    return Scatter(Ray(hit.p, wi), vec3(1f));
}

fn refract(v: vec3<f32>, n: vec3<f32>, niOverNt: f32, refractDirection: ptr<function, vec3<f32>>) -> bool {
    // ni * sin(i) = nt * sin(t)
    // sin(t) = sin(i) * (ni / nt)
    let uv = normalize(v);
    let dt = dot(uv, n);
    let discriminant = 1f - niOverNt * niOverNt * (1f - dt * dt);
    if discriminant > 0f {
        *refractDirection = normalize(niOverNt * (uv - dt * n) - sqrt(discriminant) * n);
        return true;
    }

    return false;
}

fn schlick(cosine: f32, refractionIndex: f32) -> f32 {
    var r0 = (1f - refractionIndex) / (1f + refractionIndex);
    r0 = r0 * r0;
    return r0 + pow((1f - r0) * (1f - cosine), 5f);
}

fn scatterCheckerboard(hit: Intersection, texture1: TextureDescriptor, texture2: TextureDescriptor, rngState: ptr<function, u32>) -> Scatter {
    let sines = sin(5f * hit.p.x) * sin(5f * hit.p.y) * sin(5f * hit.p.z);
    if sines < 0f {
        return scatterMixtureDensity(hit, texture1, rngState);
    } else {
        return scatterMixtureDensity(hit, texture2, rngState);
    }
}

fn scatterMissingMaterial(hit: Intersection, rngState: ptr<function, u32>) -> Scatter {
    let scatterDirection = hit.n + rngNextVec3InUnitSphere(rngState);
    // An aggressive pink color to indicate an error
    let albedo = vec3(0.9921f, 0.24705f, 0.57254f);
    return Scatter(Ray(hit.p, scatterDirection), albedo);
}

fn radiance(theta: f32, gamma: f32, channel: u32) -> f32 {
    let r = skyState.radiances[channel];
    let idx = 9u * channel;
    let p0 = skyState.params[idx + 0u];
    let p1 = skyState.params[idx + 1u];
    let p2 = skyState.params[idx + 2u];
    let p3 = skyState.params[idx + 3u];
    let p4 = skyState.params[idx + 4u];
    let p5 = skyState.params[idx + 5u];
    let p6 = skyState.params[idx + 6u];
    let p7 = skyState.params[idx + 7u];
    let p8 = skyState.params[idx + 8u];

    let cosGamma = cos(gamma);
    let cosGamma2 = cosGamma * cosGamma;
    let cosTheta = abs(cos(theta));

    let expM = exp(p4 * gamma);
    let rayM = cosGamma2;
    let mieMLhs = 1.0 + cosGamma2;
    let mieMRhs = pow(1.0 + p8 * p8 - 2.0 * p8 * cosGamma, 1.5f);
    let mieM = mieMLhs / mieMRhs;
    let zenith = sqrt(cosTheta);
    let radianceLhs = 1.0 + p0 * exp(p1 / (cosTheta + 0.01));
    let radianceRhs = p2 + p3 * expM + p5 * rayM + p6 * mieM + p7 * zenith;
    let radianceDist = radianceLhs * radianceRhs;
    return r * radianceDist;
}

struct SkyState {
    params: array<f32, 27>,
    radiances: array<f32, 3>,
    sunDirection: vec4<f32>, // w: cos(angular radius)
    sunRadiance: vec4<f32>, // rgb (0: no sun), w: solid angle
    sampling: array<vec2<f32>,2048>,
};

struct SamplingParams {
    numSamplesPerPixel: u32,
    numBounces: u32,
    accumulatedSamplesPerPixel: u32,
    clearAccumulatedSamples: u32,
    denoisingEnabled: u32,
    denoisingRadius: u32,
    denoisingSigmaColor: f32,
    // Origin of the traced region: x in bits 0-15, y in bits 16-31 (0 = full frame).
    traceOrigin: u32,
    // Nonzero: real-time frames reuse lamp samples (RESTIR).
    restir: u32,
    // Frames since the view or the scene last changed (1: the first of a new view).
    still: u32,
    pad1: u32,
    pad2: u32,
}

struct Sphere {
    centerAndPad: vec4<f32>,
    radius: f32,
    materialIdx: u32,
}

struct Material {
    id: u32,
    desc1: TextureDescriptor,
    desc2: TextureDescriptor,
    x: f32,
}

struct TextureDescriptor {
    width: u32,
    height: u32,
    offset: u32,
}

fn textureLookup(desc: TextureDescriptor, u: f32, v: f32) -> vec3<f32> {
    let textureU = clamp(u, 0f, 1f);
    let textureV = 1f - clamp(v, 0f, 1f);

    let j = min(u32(textureU * f32(desc.width)), desc.width - 1u);
    let i = min(u32(textureV * f32(desc.height)), desc.height - 1u);
    let idx = i * desc.width + j;

    let elem = textures[desc.offset + idx];
    return vec3(elem[0u], elem[1u], elem[2u]);
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>
}

struct Scatter {
    ray: Ray,
    throughput: vec3<f32>,
}

struct Intersection {
    p: vec3<f32>,
    n: vec3<f32>,
    u: f32,
    v: f32,
    t: f32,
    materialIdx: u32,
    sphereIdx: u32,
    uv1: vec2<f32>,
    geometricNormal: vec3<f32>,
}

fn rayIntersectSphere(ray: Ray, sphereIdx: u32, tmin: f32, tmax: f32, hit: ptr<function, Intersection>) -> bool {
    let sphere = spheres[sphereIdx];
    let oc = ray.origin - sphere.centerAndPad.xyz;
    let a = dot(ray.direction, ray.direction);
    let b = dot(oc, ray.direction);
    let c = dot(oc, oc) - sphere.radius * sphere.radius;
    let discriminant = b * b - a * c;

    if discriminant > 0f {
        var t = (-b - sqrt(b * b - a * c)) / a;
        if t < tmax && t > tmin {
            *hit = sphereIntersection(ray, sphere, sphereIdx, t);
            return true;
        }

        t = (-b + sqrt(b * b - a * c)) / a;
        if t < tmax && t > tmin {
            *hit = sphereIntersection(ray, sphere, sphereIdx, t);
            return true;
        }
    }

    return false;
}

fn sphereIntersection(ray: Ray, sphere: Sphere, sphereIdx: u32, t: f32) -> Intersection {
    let p = rayPointAtParameter(ray, t);
    let n = (1f / sphere.radius) * (p - sphere.centerAndPad.xyz);
    let theta = acos(-n.y);
    let phi = atan2(-n.z, n.x) + PI;
    let u = 0.5 * FRAC_1_PI * phi;
    let v = FRAC_1_PI * theta;

    // TODO: passing sphereIdx in here just to pass it to Intersection
    return Intersection(p, n, u, v, t, sphere.materialIdx, sphereIdx, vec2(0.), n);
}

fn rayPointAtParameter(ray: Ray, t: f32) -> vec3<f32> {
    return ray.origin + t * ray.direction;
}

struct Camera {
    eye: vec3<f32>,
    horizontal: vec3<f32>,
    vertical: vec3<f32>,
    u: vec3<f32>,
    v: vec3<f32>,
    lensRadius: f32,
    lowerLeftCorner: vec3<f32>,
}

fn cameraMakeRay(camera: Camera, rngState: ptr<function, u32>, u: f32, v: f32) -> Ray {
    let randomPointInLens = camera.lensRadius * rngNextVec3InUnitDisk(rngState);
    let lensOffset = randomPointInLens.x * camera.u + randomPointInLens.y * camera.v;

    let origin = camera.eye + lensOffset;
    let direction = camera.lowerLeftCorner + u * camera.horizontal + v * camera.vertical - origin;

    return Ray(origin, direction);
}

fn rngNextInCosineWeightedHemisphere(state: ptr<function, u32>) -> vec3<f32> {
    let r1 = rngNextFloat(state);
    let r2 = rngNextFloat(state);
    let sqrt_r2 = sqrt(r2);

    let z = sqrt(1f - r2);
    let phi = 2f * PI * r1;
    let x = cos(phi) * sqrt_r2;
    let y = sin(phi) * sqrt_r2;

    return vec3<f32>(x, y, z);
}

fn rngNextInUnitHemisphere(state: ptr<function, u32>) -> vec3<f32> {
    let r1 = rngNextFloat(state);
    let r2 = rngNextFloat(state);

    let phi = 2f * PI * r1;
    let sinTheta = sqrt(1f - r2 * r2);

    let x = cos(phi) * sinTheta;
    let y = sin(phi) * sinTheta;
    let z = r2;

    return vec3(x, y, z);
}

fn rngNextVec3InUnitDisk(state: ptr<function, u32>) -> vec3<f32> {
    // Generate numbers uniformly in a disk:
    // https://stats.stackexchange.com/a/481559

    // r^2 is distributed as U(0, 1).
    let r = sqrt(rngNextFloat(state));
    let alpha = 2f * PI * rngNextFloat(state);

    let x = r * cos(alpha);
    let y = r * sin(alpha);

    return vec3(x, y, 0f);
}

fn rngNextVec3InUnitSphere(state: ptr<function, u32>) -> vec3<f32> {
    let r = pow(rngNextFloat(state), 0.33333f);
    let cosTheta = 1f - 2f * rngNextFloat(state);
    let sinTheta = sqrt(1f - cosTheta * cosTheta);
    let phi = 2f * PI * rngNextFloat(state);

    let x = r * sinTheta * cos(phi);
    let y = r * sinTheta * sin(phi);
    let z = cosTheta;

    return vec3(x, y, z);
}

fn rngNextUintInRange(state: ptr<function, u32>, min: u32, max: u32) -> u32 {
    let x = rngNextInt(state);
    return min + (x) % (max - min);
}

fn blueNoiseNext() -> f32 {
    let d = blueNoiseDim;
    blueNoiseDim += 1u;
    let offset = vec2<u32>(fract(vec2(0.7548776662, 0.569840291) * f32(d / 2u + 1u)) * 64.);
    let p = (blueNoisePixel + offset) % vec2(64u);
    let mask = unpack2x16unorm(meshData[meshData[14u] + p.y * 64u + p.x]);
    return fract(select(mask.x, mask.y, (d & 1u) == 1u) + blueNoiseShift);
}

fn rngNextFloat(state: ptr<function, u32>) -> f32 {
    if BLUE_NOISE_PATH && blueNoiseDim < BLUE_NOISE_DIMS { return min(blueNoiseNext(), 0.99999994); }
    let x = rngNextInt(state);
    return f32(x) / f32(0xffffffffu);
}

fn initRng(pixel: vec2<u32>, resolution: vec2<u32>, frame: u32) -> u32 {
    // Adapted from https://github.com/boksajak/referencePT
    let seed = dot(pixel, vec2<u32>(1u, resolution.x)) ^ jenkinsHash(frame);
    return jenkinsHash(seed);
}

fn rngNextInt(state: ptr<function, u32>) -> u32 {
    // PCG random number generator
    // Based on https://www.shadertoy.com/view/XlGcRh
    let newState = *state * 747796405u + 2891336453u;
    *state = newState;
    let word = ((newState >> ((newState >> 28u) + 4u)) ^ newState) * 277803737u;
    return (word >> 22u) ^ word;
}

fn jenkinsHash(input: u32) -> u32 {
    var x = input;
    x += x << 10u;
    x ^= x >> 6u;
    x += x << 3u;
    x ^= x >> 11u;
    x += x << 15u;
    return x;
}

// Atmosphere plus sun disk, for unit directions.
fn skyRadiance(v: vec3<f32>) -> vec3<f32> {return atmosphereRadiance(v)+sunRadiance(v);}
fn atmosphereRadiance(v: vec3<f32>) -> vec3<f32> {
    let theta=acos(clamp(v.y,-1.,1.));
    let gamma=acos(clamp(dot(v,skyState.sunDirection.xyz),-1.,1.));
    return vec3(radiance(theta,gamma,0u),radiance(theta,gamma,1u),radiance(theta,gamma,2u));
}
// Uniform direction in the sun's cone, and its solid-angle pdf.
fn sampleSun(rng: ptr<function,u32>) -> vec3<f32> {
    let width=skyState.sunRadiance.w/(2.*PI); // 1 - cos(angular radius)
    let z=1.-rngNextFloat(rng)*width;let phi=2.*PI*rngNextFloat(rng);
    let radial=sqrt(max(0.,1.-z*z));
    return pixarOnb(skyState.sunDirection.xyz)*vec3(radial*cos(phi),radial*sin(phi),z);
}
fn sunPdf(v: vec3<f32>) -> f32 {
    return select(0.,1./skyState.sunRadiance.w,dot(v,skyState.sunDirection.xyz)>=skyState.sunDirection.w);
}
fn sunRadiance(v: vec3<f32>) -> vec3<f32> {
    return select(vec3(0.),skyState.sunRadiance.rgb,dot(v,skyState.sunDirection.xyz)>=skyState.sunDirection.w);
}
fn misWeight(a: f32,b: f32) -> f32 {return a*a/max(a*a+b*b,1e-30);}
fn skyPdf(direction: vec3<f32>) -> f32 {
    let row=min(u32((1.-clamp(direction.y,-1.,1.))*16.),31u);
    let phi=atan2(direction.z,direction.x);
    let column=min(u32(fract(phi/(2.*PI)+1.)*64.),63u);
    return skyState.sampling[row*64u+column].y;
}
fn sampleSky(rng: ptr<function,u32>) -> vec3<f32> {
    let quantile=min(rngNextFloat(rng),0.99999994);
    var lo=0u;var hi=2047u;
    while lo<hi {
        let middle=(lo+hi)/2u;
        if quantile<skyState.sampling[middle].x {hi=middle;} else {lo=middle+1u;}
    }
    let y=1.-2.*(f32(lo/64u)+rngNextFloat(rng))/32.;
    let phi=2.*PI*(f32(lo%64u)+rngNextFloat(rng))/64.;
    let r=sqrt(max(0.,1.-y*y));
    return vec3(r*cos(phi),y,r*sin(phi));
}
fn skyOccluded(ray: Ray) -> bool {return occludedWithin(ray,MAX_T);}
// Any blocker along a unit-direction ray before distance `limit`.
fn occludedWithin(ray: Ray, limit: f32) -> bool {
    let diffuse=pathDiffuse;if OPACITY_APPROX {pathDiffuse=false;}
    // Traversal reports missed boxes at MAX_T: a larger limit would visit every node.
    let blocked=skyOccludedExact(ray,min(limit,MAX_T));
    if OPACITY_APPROX {pathDiffuse=diffuse;}
    return blocked;
}
fn skyOccludedExact(ray: Ray, limit: f32) -> bool {
    shadowRay=true;
    let blocked=skyOccludedQuery(ray,limit);
    shadowRay=false;
    return blocked;
}
fn skyOccludedQuery(ray: Ray, limit: f32) -> bool {
    if ALPHA_CAP {alphaQueries=0u;}
    for(var i=0u;i<SPHERE_COUNT;i+=1u) {
        var hit=Intersection();
        if rayIntersectSphere(ray,i,MIN_T,limit,&hit) {return true;}
    }
    return TRACE_MESHES && meshOccluded(ray,limit);
}

// Lamps: emissive scene triangles, chosen by an alias table over weight × area
// (mesh.rs lamp_table); a uniform point on the chosen triangle.
struct LampSample {direction: vec3<f32>, distance: f32, radiance: vec3<f32>, pdf: f32}
// Solid-angle pdf of reaching a lamp point: selection (weight × area / total) times
// the uniform area pdf (1 / area) times the area-to-solid-angle factor. Area cancels.
fn lampPdfAt(material: u32, distance: f32, cosLight: f32) -> f32 {
    let table=meshData[12u];
    if table==0u {return 0.;}
    let weight=meshFloat(meshMaterialOffset(material)+12u);
    return weight/meshFloat(table+1u)*distance*distance/max(cosLight,1e-8);
}
// A lamp chosen among LAMP_CANDIDATES by target = weight × max(n·ω, 0) (cheap: no
// texture, no BSDF, no visibility). `pdf` is the chosen point's source pdf (for MIS),
// `scale` the RIS estimator weight (1/pdf when there is one candidate); 0 = none.
struct LampChoice {direction: vec3<f32>, distance: f32, radiance: vec3<f32>, pdf: f32, scale: f32}
fn lampRis(origin: vec3<f32>, normal: vec3<f32>, candidates: u32, rng: ptr<function,u32>) -> LampChoice {
    var chosen=LampPoint();var chosenTarget=0.;var sum=0.;
    for(var k=0u;k<candidates;k+=1u) {
        let candidate=sampleLampPoint(origin,rng);
        let score=candidate.weight*max(dot(normal,candidate.direction),0.);
        let w=select(0.,score/candidate.pdf,candidate.pdf>0.);
        sum+=w;
        if w>0. && rngNextFloat(rng)*sum<w {chosen=candidate;chosenTarget=score;}
    }
    if chosenTarget<=0. {return LampChoice();}
    return LampChoice(chosen.direction,chosen.distance,lampRadiance(chosen),chosen.pdf,sum/(f32(candidates)*chosenTarget));
}
// `lamp` is the lamp's index in the lamp table (mesh.rs lamp_table).
struct LampPoint {direction: vec3<f32>, distance: f32, pdf: f32, material: u32, lamp: u32, bary: vec2<f32>, cosLight: f32, weight: f32}
fn lampRadiance(point: LampPoint) -> vec3<f32> {
    let m=meshMaterialOffset(point.material);
    let emission=meshVec3(m+4u);
    if meshData[m+32u]==0u {return emission;}
    let tri=meshData[2u]+meshData[meshData[meshData[12u]+2u]+point.lamp*16u+11u]*4u;
    let va=vertexOffset(meshData[tri]);let vb=vertexOffset(meshData[tri+1u]);let vc=vertexOffset(meshData[tri+2u]);
    let w=vec3(1.-point.bary.x-point.bary.y,point.bary.x,point.bary.y);
    let uv0=w.x*vertexUV0(va)+w.y*vertexUV0(vb)+w.z*vertexUV0(vc);
    let uv1=w.x*vertexUV1(va)+w.y*vertexUV1(vb)+w.z*vertexUV1(vc);
    let cones=coneActive;coneActive=false;
    let radiance=emission*meshTexture(m+32u,uv0,uv1,true).rgb;
    coneActive=cones;
    return radiance;
}
fn sampleLamp(origin: vec3<f32>, rng: ptr<function,u32>) -> LampSample {
    let point=sampleLampPoint(origin,rng);
    return LampSample(point.direction,point.distance,lampRadiance(point),point.pdf);
}
fn sampleLampPoint(origin: vec3<f32>, rng: ptr<function,u32>) -> LampPoint {
    let table=meshData[12u];let count=meshData[table];
    let x=rngNextFloat(rng)*f32(count);let slot=min(u32(x),count-1u);
    let entry=table+3u+slot*2u;
    let pick=select(meshData[entry+1u],slot,fract(x)<meshFloat(entry));
    let r=sqrt(rngNextFloat(rng));let s=rngNextFloat(rng);
    return lampAt(pick,vec2(r*(1.-s),r*s),origin);
}
// The point with barycentrics (b1, b2) on lamp `lamp`, seen from `origin`: one 64-byte
// record (corner + material, edge + weight, edge + triangle, unit normal).
fn lampAt(lamp: u32, bary: vec2<f32>, origin: vec3<f32>) -> LampPoint {
    let table=meshData[12u];
    let record=meshData[table+2u]+lamp*16u;
    let a=meshVec4(record);let e1=meshVec4(record+4u);let e2=meshVec3(record+8u);let normal=meshVec3(record+12u);
    let toLight=a.xyz+bary.x*e1.xyz+bary.y*e2-origin;
    let distance=length(toLight);let direction=toLight/max(distance,1e-20);
    let cosLight=abs(dot(normal,direction));
    // Source pdf in solid angle: selection (weight x area / total) x the uniform area
    // pdf (1 / area) x the area-to-solid-angle factor. Texture coordinates are fetched
    // only for the winner (lampRadiance).
    let pdf=e1.w/meshFloat(table+1u)*distance*distance/max(cosLight,1e-8);
    return LampPoint(direction,distance,pdf,bitcast<u32>(a.w),lamp,bary,cosLight,e1.w);
}

// ReSTIR in area measure: target = lamp weight x max(n.w, 0) x cosLight / d^2 (no
// BSDF, texture or visibility); the source pdf of a fresh candidate is weight / total.
fn restirTarget(point: LampPoint, normal: vec3<f32>) -> f32 {
    return point.weight*max(dot(normal,point.direction),0.)*point.cosLight/max(point.distance*point.distance,1e-12);
}
struct Reservoir {lamp: u32, bary: vec2<f32>, w: f32, m: f32, depth: f32, normal: vec3<f32>}
fn reservoirSlot(parity: u32, pixel: u32) -> u32 {return (parity*frameData.x*frameData.y+pixel)*3u;}
// DI_SHARED final reservoirs keep their shadow-ray result in the lamp word's top bit.
const RESERVOIR_VISIBLE = 0x80000000u;
fn loadReservoir(slot: u32) -> Reservoir {
    let a=reservoirs[slot];let b=reservoirs[slot+1u];let c=reservoirs[slot+2u];
    return Reservoir(a.x&0x7fffffffu,unpack2x16unorm(a.y),bitcast<f32>(b.x),bitcast<f32>(b.y),bitcast<f32>(c.x),decodeOct(c.y));
}
fn storeReservoir(slot: u32, r: Reservoir) {
    reservoirs[slot]=vec2(r.lamp,pack2x16unorm(r.bary));
    reservoirs[slot+1u]=vec2(bitcast<u32>(r.w),bitcast<u32>(r.m));
    reservoirs[slot+2u]=vec2(bitcast<u32>(r.depth),pack2x16snorm(encodeOct(r.normal)));
}
// ReSTIR DI for lamps at the primary surface, in two passes per frame.
// csRestirInitial (before the path tracer): fresh RIS candidates plus the pixel's
// final reservoir of the previous frame (followed along the motion), the winner
// shadow-tested (W = 0 when occluded, M still counts) and stored as this frame's
// initial reservoir. restirLamps (the path tracer's primary surface): the pixel's own
// and RESTIR_SPATIAL neighbours' initial reservoirs of this frame, merged with
// M-capped, 1/M-normalised weights (biased, the usual real-time variant), one shadow
// ray for the winner, stored as the final reservoir for the next frame.
const RESTIR_INITIAL = 2u;
// Fresh candidates for one pixel per 2x2 block (rotating each frame); the spatial pass
// reads a pixel's block. Off (RT_RESTIR_FULL): every pixel.
override RESTIR_BLOCKS: bool = true;
fn restirBlock(x: u32, y: u32) -> u32 {
    if !RESTIR_BLOCKS {return frameData.x*y+x;}
    return (frameData.x+1u)/2u*(y/2u)+x/2u;
}
fn reservoirSet(group: u32, pixel: u32) -> u32 {return reservoirSlot(group,pixel);}
struct RestirMerge {chosen: LampPoint, score: f32, sum: f32, m: f32}
fn restirAdd(merge: ptr<function,RestirMerge>, r: Reservoir, origin: vec3<f32>, n: vec3<f32>, cap: f32, rng: ptr<function,u32>) {
    let candidate=lampAt(r.lamp,r.bary,origin);
    let score=restirTarget(candidate,n);
    let rm=min(r.m,cap);
    let w=score*r.w*rm;
    (*merge).sum+=w;(*merge).m+=rm;
    if w>0. && rngNextFloat(rng)*(*merge).sum<w {(*merge).chosen=candidate;(*merge).score=score;}
}
// The reservoirs resample a target without visibility, so on their own they keep
// proposing bright lamps behind walls (night Bistro: 72% of lit pixels got no lamp light
// in a frame). With RESTIR_VISIBILITY (off: RT_RESTIR_VIS=0):
// - the winner of the fresh candidates gets a shadow ray; hidden, the candidates enter
//   the history as LAMP_CANDIDATES samples that found nothing;
// - a final winner that is hidden at the shaded pixel is forgotten, and the pixel starts
//   over from fresh candidates (kept with weight 0 and its full history length it
//   dilutes them: 3-10% darker, 38% per 8x8 block in night Bistro).
// Night Bistro: noise per pixel 135% -> 34%, per 8x8 block 31% -> 12%; the mean of 16
// frames is as close to the reference as before.
override RESTIR_VISIBILITY: bool = true;
fn restirVisible(merge: RestirMerge, hit: Intersection) -> bool {
    return merge.score>0. && !occludedWithin(Ray(hit.p+hit.geometricNormal*0.0001,merge.chosen.direction),merge.chosen.distance*0.999);
}
// The merged winner's reservoir (W = 0 if none, or if `visible` is false).
fn restirResolve(merge: RestirMerge, hit: Intersection, cap: f32, depth: f32, visible: bool) -> Reservoir {
    var stored=Reservoir(merge.chosen.lamp,merge.chosen.bary,0.,min(merge.m,cap),depth,hit.n);
    if merge.score>0. && visible {stored.w=merge.sum/(merge.m*merge.score);}
    return stored;
}
@compute @workgroup_size(8, 8)
fn csRestirInitial(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    // Traversal with the workgroup-memory stack indexes its slots by lane (64 lanes).
    quadLane = lane;
    // With blocks, thread (x, y) samples block (x, y)'s pixel for this frame.
    let turn=samplingParams.restir&3u;
    let at=select(id.xy,id.xy*2u+vec2(turn&1u,turn>>1u),RESTIR_BLOCKS);
    if !RESTIR || at.x>=frameData.x || at.y>=frameData.y {return;}
    let pixel=frameData.x*at.y+at.x;
    let slot=select(pixel,restirBlock(at.x,at.y),RESTIR_BLOCKS);
    var rng=initRng(at,frameData.xy,frameData.z^0x5bd1e995u);
    var jitter=vec2(0.5);
    if SHARED_JITTER {jitter=frameJitter(frameData.z);}
    // With the visibility raster, the primary hit comes from it (same jitter).
    if VIS_PRIMARY {jitter=primaryJitter();gbufferPixel=pixel;visibilityPending=true;}
    gbufferSample=vec2<f32>(at)+jitter;
    let u=(f32(at.x)+jitter.x)/f32(frameData.x);let v=(f32(at.y)+jitter.y)/f32(frameData.y);
    if RAY_CONES {coneActive=false;}
    let ray=cameraMakeRay(camera,&rng,u,1.-v);
    var hit=Intersection();
    var stored=Reservoir();
    if intersection(ray,&hit) && hit.sphereIdx==0xffffffffu {
        let n=hit.n;
        var merge=RestirMerge();
        let cap=RESTIR_CAP*f32(LAMP_CANDIDATES);
        // Temporal: the pixel's final reservoir of the previous frame.
        var history=Reservoir();var remembered=false;
        let previous=previousPixel(hit.p);
        if previous.x>=0. && previous.y>=0. && previous.x<f32(frameData.x) && previous.y<f32(frameData.y) {
            let seenFrom=distance(prevCamera.eye,hit.p);
            let last=select(u32(previous.y)*frameData.x+u32(previous.x),restirBlock(u32(previous.x),u32(previous.y)),DI_SHARED);
            history=loadReservoir(reservoirSet((samplingParams.restir&1u)^1u,last));
            remembered=history.m>0. && abs(history.depth-seenFrom)<=0.1*seenFrom && dot(history.normal,n)>=0.9;
        }
        let total=meshFloat(meshData[12u]+1u);
        for(var k=0u;k<LAMP_CANDIDATES;k+=1u) {
            let candidate=sampleLampPoint(hit.p,&rng);
            let score=restirTarget(candidate,n);
            let w=score*total/max(candidate.weight,1e-30);
            merge.sum+=w;
            if w>0. && rngNextFloat(&rng)*merge.sum<w {merge.chosen=candidate;merge.score=score;}
        }
        merge.m=f32(LAMP_CANDIDATES);
        // Fresh candidates whose winner is hidden bring nothing into the history
        // (they still count as LAMP_CANDIDATES samples).
        if RESTIR_VISIBILITY && merge.score>0. && !restirVisible(merge,hit) {
            merge.chosen=LampPoint();merge.score=0.;merge.sum=0.;
        }
        if remembered {restirAdd(&merge,history,hit.p,n,cap,&rng);}
        stored=restirResolve(merge,hit,cap,distance(camera.eye,hit.p),true);
    }
    storeReservoir(reservoirSet(RESTIR_INITIAL,slot),stored);
}
// This frame's first-pass reservoirs of the block at `here` and RESTIR_SPATIAL random
// neighbouring blocks on a similar surface, merged at `hit`.
fn restirSpatialMerge(hit: Intersection, here: vec2<f32>, rng: ptr<function,u32>) -> RestirMerge {
    let n=hit.n;
    let cap=RESTIR_CAP*f32(LAMP_CANDIDATES);
    let seenFrom=distance(camera.eye,hit.p);
    var merge=RestirMerge();
    for(var k=0u;k<=RESTIR_SPATIAL;k+=1u) {
        var at=here;
        if k>0u {
            let angle=6.2831853*rngNextFloat(rng);
            at+=RESTIR_RADIUS*sqrt(rngNextFloat(rng))*vec2(cos(angle),sin(angle));
        }
        if at.x<0. || at.y<0. || at.x>=f32(frameData.x) || at.y>=f32(frameData.y) {continue;}
        let r=loadReservoir(reservoirSet(RESTIR_INITIAL,restirBlock(u32(at.x),u32(at.y))));
        if r.m<=0. || abs(r.depth-seenFrom)>0.1*seenFrom || dot(r.normal,n)<0.9 {continue;}
        restirAdd(&merge,r,hit.p,n,cap,rng);
    }
    return merge;
}
fn restirShade(point: LampPoint, surface: PbrSurface, hit: Intersection, w: f32) -> vec3<f32> {
    let eval=evaluatePbr(surface,hit,point.direction);
    return lampRadiance(point)*eval.rgb*point.cosLight/max(point.distance*point.distance,1e-12)*w;
}
// DI_SHARED (with RESTIR_BLOCKS): the spatial merge and the shadow ray run once per 2x2
// block, at the block's first-pass pixel (csRestirShared); pixels on a similar surface
// shade the block's sample with its weight and visibility, others (edges) merge and
// trace their own. Opt-in (mod.rs di_shared: faster, 2x2-correlated noise).
override DI_SHARED: bool = false;
fn restirLamps(hit: Intersection, surface: PbrSurface, rng: ptr<function,u32>) -> vec3<f32> {
    let cap=RESTIR_CAP*f32(LAMP_CANDIDATES);
    let seenFrom=distance(camera.eye,hit.p);
    let here=vec2(f32(gbufferPixel%frameData.x),f32(gbufferPixel/frameData.x))+vec2(0.5);
    if DI_SHARED {
        let slot=reservoirSet(samplingParams.restir&1u,restirBlock(gbufferPixel%frameData.x,gbufferPixel/frameData.x));
        let r=loadReservoir(slot);
        if r.m>0. && abs(r.depth-seenFrom)<=0.1*seenFrom && dot(r.normal,hit.n)>=0.9 {
            if (reservoirs[slot].x&RESERVOIR_VISIBLE)==0u || r.w<=0. {return vec3(0.);}
            return restirShade(lampAt(r.lamp,r.bary,hit.p),surface,hit,r.w);
        }
        // An edge: this pixel's own merge and shadow ray (nothing stored).
        let merge=restirSpatialMerge(hit,here,rng);
        if !restirVisible(merge,hit) {return vec3(0.);}
        return restirShade(merge.chosen,surface,hit,restirResolve(merge,hit,cap,seenFrom,true).w);
    }
    let merge=restirSpatialMerge(hit,here,rng);
    let visible=restirVisible(merge,hit);
    var stored=restirResolve(merge,hit,cap,seenFrom,true);
    // A hidden winner is not kept: the pixel starts over from fresh candidates.
    if RESTIR_VISIBILITY && !visible {stored.w=0.;stored.m=0.;}
    storeReservoir(reservoirSet(samplingParams.restir&1u,gbufferPixel),stored);
    if stored.w<=0. || !visible {return vec3(0.);}
    return restirShade(merge.chosen,surface,hit,stored.w);
}
@compute @workgroup_size(8, 8)
fn csRestirShared(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    quadLane = lane;
    let turn=samplingParams.restir&3u;
    let at=id.xy*2u+vec2(turn&1u,turn>>1u);
    if !RESTIR || !DI_SHARED || at.x>=frameData.x || at.y>=frameData.y {return;}
    let pixel=frameData.x*at.y+at.x;
    var rng=initRng(at,frameData.xy,frameData.z^0x1b873593u);
    var jitter=vec2(0.5);
    if SHARED_JITTER {jitter=frameJitter(frameData.z);}
    if VIS_PRIMARY {jitter=primaryJitter();gbufferPixel=pixel;visibilityPending=true;}
    gbufferSample=vec2<f32>(at)+jitter;
    let u=(f32(at.x)+jitter.x)/f32(frameData.x);let v=(f32(at.y)+jitter.y)/f32(frameData.y);
    if RAY_CONES {coneActive=false;}
    let ray=cameraMakeRay(camera,&rng,u,1.-v);
    var hit=Intersection();
    var stored=Reservoir();
    var flags=0u;
    if intersection(ray,&hit) && hit.sphereIdx==0xffffffffu {
        let merge=restirSpatialMerge(hit,vec2<f32>(at)+vec2(0.5),&rng);
        let cap=RESTIR_CAP*f32(LAMP_CANDIDATES);
        // History keeps the visibility-free weight; the shadow ray's result is a flag.
        stored=restirResolve(merge,hit,cap,distance(camera.eye,hit.p),true);
        flags=select(0u,RESERVOIR_VISIBLE,restirVisible(merge,hit));
    }
    let slot=reservoirSet(samplingParams.restir&1u,restirBlock(at.x,at.y));
    storeReservoir(slot,stored);
    reservoirs[slot].x|=flags;
}

// ReSTIR GI reuse (Ouyang et al. 2021), after the path tracer. The pixel's own sample,
// its final reservoir of the previous frame (followed along the motion) and
// GI_SPATIAL neighbours' samples of this frame are merged with M-capped,
// 1/M-normalised weights; reused samples carry the reconnection Jacobian. Target:
// luminance of BSDF x cos x radiance at the primary point (material from the G-buffer).
// The winner, when borrowed, gets a visibility ray; its contribution is added to the
// pixel and kept as the final reservoir for the next frame.
struct GiSample {primary: vec3<f32>, primaryNormal: vec3<f32>, secondary: vec3<f32>, secondaryNormal: vec3<f32>, radiance: vec3<f32>, pdf: f32}
fn loadGiSample(pixel: u32) -> GiSample {
    let a=giData[pixel*3u];let b=giData[pixel*3u+1u];let c=giData[pixel*3u+2u];
    return GiSample(bitcast<vec3<f32>>(a.xyz),decodeOct(a.w),bitcast<vec3<f32>>(b.xyz),decodeOct(b.w),
        vec3(unpack2x16float(c.x),unpack2x16float(c.y).x),select(0.,bitcast<f32>(c.z),c.w==1u));
}
fn giFinalSlot(parity: u32, pixel: u32) -> u32 {return frameData.x*frameData.y*3u+(parity*frameData.x*frameData.y+pixel)*2u;}
struct GiPick {secondary: vec3<f32>, secondaryNormal: vec3<f32>, radiance: vec3<f32>, score: f32, borrowed: bool}
fn giTarget(surface: PbrSurface, hit: Intersection, secondary: vec3<f32>, radiance: vec3<f32>) -> f32 {
    let eval=evaluatePbr(surface,hit,normalize(secondary-hit.p));
    return dot(eval.rgb*radiance,vec3(0.2126,0.7152,0.0722));
}
// Reconnecting a sample made at `source` to the primary point `destination`: ratio of the solid
// angles the secondary point's area subtends (clamped against extreme reuse).
fn giJacobian(secondary: vec3<f32>, secondaryNormal: vec3<f32>, source: vec3<f32>, destination: vec3<f32>) -> f32 {
    let a=source-secondary;let b=destination-secondary;
    let cosFrom=abs(dot(secondaryNormal,normalize(a)));let cosTo=abs(dot(secondaryNormal,normalize(b)));
    let j=(cosTo*dot(a,a))/max(cosFrom*dot(b,b),1e-12);
    return select(j,0.,j>10. || j<0.1);
}
fn giCap() -> f32 {
    if !GI_STILL_GROWTH {return GI_CAP;}
    return GI_CAP+f32(max(samplingParams.still,1u)-1u)*select(1.,0.25,GI_BLOCKS);
}
@compute @workgroup_size(8, 8)
fn csRestirGi(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    // Traversal with the workgroup-memory stack indexes its slots by lane (64 lanes).
    quadLane = lane;
    if !RESTIR_GI || id.x>=frameData.x || id.y>=frameData.y {return;}
    let pixel=frameData.x*id.y+id.x;
    let parity=samplingParams.restir&1u;
    let own=loadGiSample(pixel);
    // Records: 0 none, 1 a sample, 2 the primary point only (GI_BLOCKS), 3 a null
    // sample (GI_NULL_SAMPLES).
    let state=giData[pixel*3u+2u].w;
    if state==0u {
        giData[giFinalSlot(parity,pixel)]=vec4(0u);giData[giFinalSlot(parity,pixel)+1u]=vec4(0u);
        return;
    }
    var rng=initRng(id.xy,frameData.xy,frameData.z^0x27d4eb2du);
    let g0=gbuffer[pixel*3u];let g1=gbuffer[pixel*3u+1u];let g2=gbuffer[pixel*3u+2u];
    var hit=Intersection();hit.p=own.primary;hit.n=own.primaryNormal;hit.geometricNormal=own.primaryNormal;
    let rough=max(g1.w,0.045);let alpha=rough*rough;let base=g0.xyz;let metal=g2.z;
    let view=normalize(camera.eye-own.primary);
    let f0=mix(vec3(0.04),base,metal);
    let surface=PbrSurface(base,metal,alpha*alpha,view,max(dot(hit.n,view),1e-6),f0,clamp(max(f0.r,max(f0.g,f0.b)),0.25,0.9),0.,1.5,true,vec3(0.),1.);
    // The pixel's own sample: W = 1 / pdf.
    var pick=GiPick();var sum=0.;var m=0.;var pickTemporal=false;
    if state==1u {
        pick=GiPick(own.secondary,own.secondaryNormal,own.radiance,giTarget(surface,hit,own.secondary,own.radiance),false);
        sum=pick.score/own.pdf;m=1.;
    } else if state==3u {m=1.;}
    // Temporal: last frame's final reservoir along the motion.
    var disoccluded=true;
    let previous=previousPixel(own.primary);
    if previous.x>=0. && previous.y>=0. && previous.x<f32(frameData.x) && previous.y<f32(frameData.y) {
        let slot=giFinalSlot(parity^1u,u32(previous.y)*frameData.x+u32(previous.x));
        let a=giData[slot];let b=giData[slot+1u];
        let rm=min(unpack2x16float(b.y).y,giCap());
        let seenFrom=distance(prevCamera.eye,own.primary);
        // The same surface was on screen last frame (whatever became of its history).
        disoccluded=abs(bitcast<f32>(b.w)-seenFrom)>0.1*seenFrom;
        if rm>0. && !disoccluded {
            let secondary=bitcast<vec3<f32>>(a.xyz);let radiance=vec3(unpack2x16float(b.x),unpack2x16float(b.y).x);
            let score=giTarget(surface,hit,secondary,radiance);
            let stored=bitcast<f32>(b.z);
            // A sample that carries light, but none to this point (the pixel moved over
            // the surface, or its jittered position landed on another texel of a
            // normal map, and the sample is now below its horizon), is unusable here
            // and dropped. Kept with weight 0, its whole history counts as samples
            // that found nothing and the weight can never come back: after 120 frames
            // of a turning camera Sponza was 23% darker than without reuse, day Bistro
            // 11%, the living room 9% (with this and `moved` below: 5%, 2%, 6%).
            if !(GI_DROP_UNUSABLE && stored>0. && score<=0.) {
                let w=score*stored*rm;
                sum+=w;m+=rm;
                if w>0. && rngNextFloat(&rng)*sum<w {pick=GiPick(secondary,decodeOct(a.w),radiance,score,true);pickTemporal=true;}
            }
        }
    }
    // Spatial: this frame's samples of nearby pixels on a similar surface.
    let here=vec2(f32(id.x),f32(id.y))+vec2(0.5);
    let depth=distance(camera.eye,own.primary);
    // Neighbours only while the pixel's own history is short (after a disocclusion):
    // with a long history they add nothing measurable, and each costs a scattered read.
    // Also while the pixel moves over the surface (by more than GI_MOVING pixels a
    // frame): every step to another point loses the light that point sees and the
    // sample's first owner could not sample, and only fresh samples bring it back.
    // (The G-buffer's motion is measured from this frame's sample position, so the
    // sub-pixel jitter does not count as movement.)
    let moved=GI_MOVING>0. && !disoccluded && length(g2.xy)>GI_MOVING;
    let neighbours=select(0u,GI_SPATIAL,m<GI_SPATIAL_BELOW || moved);
    // A newly visible pixel without a sample of its own this frame looks at the pixels
    // that hold this frame's samples, one per 2x2 block with GI_BLOCKS, starting with
    // its own block's. Random pixels hold a sample one time in four, which left such
    // pixels dark (20-28% under camera motion). Not for a pixel whose history was
    // forgotten because its pick was hidden: refilling those brightens the image.
    let turn=(samplingParams.restir+2u)&3u;
    let empty=m<=0. && disoccluded;
    for(var k=0u;k<neighbours;k+=1u) {
        var at=here;
        if k>0u || !empty {
            let angle=6.2831853*rngNextFloat(&rng);
            at+=GI_RADIUS*sqrt(rngNextFloat(&rng))*vec2(cos(angle),sin(angle));
        }
        if at.x<0. || at.y<0. || at.x>=f32(frameData.x) || at.y>=f32(frameData.y) {continue;}
        var holder=vec2<u32>(at);
        if GI_BLOCKS && empty {holder=(holder&vec2(0xfffffffeu))|vec2(turn&1u,turn>>1u);}
        if any(holder>=frameData.xy) || all(holder==id.xy) {continue;}
        let index=holder.y*frameData.x+holder.x;
        let other=loadGiSample(index);
        let nothing=giData[index*3u+2u].w==3u;
        if (other.pdf<=0. && !nothing) || abs(distance(camera.eye,other.primary)-depth)>0.1*depth || dot(other.primaryNormal,own.primaryNormal)<0.9 {continue;}
        // A neighbour's null sample: one more sample that found nothing.
        if nothing {m+=1.;continue;}
        let jacobian=giJacobian(other.secondary,other.secondaryNormal,other.primary,own.primary);
        // Outside the Jacobian's range the sample is not reusable here at all.
        if jacobian<=0. && GI_DROP_UNUSABLE {continue;}
        let score=giTarget(surface,hit,other.secondary,other.radiance);
        let w=score*jacobian/other.pdf;
        sum+=w;m+=1.;
        if w>0. && rngNextFloat(&rng)*sum<w {pick=GiPick(other.secondary,other.secondaryNormal,other.radiance,score,true);pickTemporal=false;}
    }
    var weight=select(0.,sum/(m*pick.score),pick.score>0.);
    let toSecondary=pick.secondary-own.primary;let span=length(toSecondary);
    // Degenerate or non-finite picks contribute nothing (a NaN ray would traverse the
    // whole BVH: every box test passes).
    if !(span>1e-4 && span<1e6 && weight<1e30) {weight=0.;}
    var visible=true;
    let recheck=((id.x&1u)+2u*(id.y&1u)+samplingParams.restir)%4u==0u;
    if pick.borrowed && weight>0. && !(GI_TEMPORAL_TRUST && GI_VISIBILITY_REUSE && pickTemporal && !recheck) {
        visible=!occludedWithin(Ray(own.primary+own.primaryNormal*0.0001,toSecondary/span),span*0.999);
    }
    let slot=giFinalSlot(parity,pixel);
    giData[slot]=vec4(bitcast<vec3<u32>>(pick.secondary),pack2x16snorm(encodeOct(pick.secondaryNormal)));
    // A hidden pick is not kept: the pixel starts over.
    if GI_DROP_UNUSABLE && !visible {m=0.;}
    giData[slot+1u]=vec4(pack2x16float(pick.radiance.rg),pack2x16float(vec2(pick.radiance.b,min(m,giCap()))),bitcast<u32>(select(weight,0.,GI_VISIBILITY_REUSE && !visible)),bitcast<u32>(depth));
    if !visible {weight=0.;}
    var toShade=toSecondary;var shadeSpan=span;
    if GI_SHADE>0u && !disoccluded {
        let ownPick=pick;let ownWeight=weight;
        var shadeSum=0.;var shadeM=0.;var found=0u;
        let seenFrom=distance(prevCamera.eye,hit.p);
        // Up to three times as many tries as neighbours wanted, the later ones closer:
        // on detailed geometry and narrow surfaces most pixels at the full radius are
        // other surfaces, and a pixel left with its own sample keeps its speckle.
        for(var k=0u;k<3u*GI_SHADE && found<GI_SHADE;k+=1u) {
            let angle=6.2831853*rngNextFloat(&rng);
            let reach=GI_SHADE_RADIUS*select(1.,select(0.4,0.15,k>=2u*GI_SHADE),k>=GI_SHADE);
            let at=previous+max(reach*sqrt(rngNextFloat(&rng)),1.5)*vec2(cos(angle),sin(angle));
            if at.x<0. || at.y<0. || at.x>=f32(frameData.x) || at.y>=f32(frameData.y) {continue;}
            let q=vec2<u32>(at);
            let other=giFinalSlot(parity^1u,q.y*frameData.x+q.x);
            let a=giData[other];let b=giData[other+1u];
            let rm=min(unpack2x16float(b.y).y,giCap());
            if rm<=0. {continue;}
            // Where that pixel's surface point was (its distance along its pixel's ray).
            let uv=(vec2<f32>(q)+vec2(0.5))/vec2<f32>(frameData.xy);
            let source=prevCamera.eye+normalize(prevCamera.lowerLeftCorner+uv.x*prevCamera.horizontal+(1.-uv.y)*prevCamera.vertical-prevCamera.eye)*bitcast<f32>(b.w);
            // The same surface: in this pixel's tangent plane, facing the same way (the
            // normal of the pixel that far from this one in the current frame).
            if abs(dot(source-hit.p,hit.n))>0.02*seenFrom || abs(bitcast<f32>(b.w)-seenFrom)>0.1*seenFrom {continue;}
            let facing=vec2<i32>(here+at-previous);
            if any(facing<vec2(0)) || any(facing>=vec2<i32>(frameData.xy)) || dot(gbuffer[(u32(facing.y)*frameData.x+u32(facing.x))*3u+1u].xyz,hit.n)<GI_SHADE_NORMAL {continue;}
            let secondary=bitcast<vec3<f32>>(a.xyz);let secondaryNormal=decodeOct(a.w);
            let jacobian=giJacobian(secondary,secondaryNormal,source,hit.p);
            if jacobian<=0. {continue;}
            let radiance=vec3(unpack2x16float(b.x),unpack2x16float(b.y).x);
            let score=giTarget(surface,hit,secondary,radiance);
            let w=score*bitcast<f32>(b.z)*jacobian*rm;
            shadeSum+=w;shadeM+=rm;found+=1u;
            if w>0. && rngNextFloat(&rng)*shadeSum<w {pick=GiPick(secondary,secondaryNormal,radiance,score,true);}
        }
        // With no usable neighbour the pixel shows its own sample, as without this.
        if shadeM>0. {
            weight=select(0.,shadeSum/(shadeM*pick.score),shadeSum>0.);
            toShade=pick.secondary-hit.p;shadeSpan=length(toShade);
            if weight>0. && (!(shadeSpan>1e-4 && shadeSpan<1e6 && weight<1e30) || occludedWithin(Ray(hit.p+hit.n*0.0001,toShade/shadeSpan),shadeSpan*0.999)) {
                // A neighbour's light this pixel cannot see: its own sample instead
                // (black would count the light the neighbours cannot see as missing
                // too: 5-7% darker overall), but no brighter than GI_SHADE_OWN_LIMIT
                // times what the neighbours found, so a standing bright sample does
                // not come back through this door.
                pick=ownPick;toShade=toSecondary;shadeSpan=span;
                weight=min(ownWeight,GI_SHADE_OWN_LIMIT*shadeSum/(shadeM*max(ownPick.score,1e-30)));
            }
        }
    }
    if weight<=0. {return;}
    let eval=evaluatePbr(surface,hit,toShade/shadeSpan);
    let indirect=eval.rgb*pick.radiance*weight*select(1.,1./f32(max(samplingParams.numSamplesPerPixel,1u)),DENOISER_INPUT);
    let c=imageBuffer[pixel];
    imageBuffer[pixel]=array<f32,3>(c[0]+indirect.r,c[1]+indirect.g,c[2]+indirect.b);
}

fn encodeOct(direction:vec3<f32>)->vec2<f32>{
    let v=direction/(abs(direction.x)+abs(direction.y)+abs(direction.z));
    if v.z>=0.{return v.xy;}
    return (vec2(1.)-abs(v.yx))*select(vec2(-1.),vec2(1.),v.xy>=vec2(0.));
}
fn decodeOct(packed:u32)->vec3<f32>{
    let p=unpack2x16snorm(packed);var n=vec3(p,1.-abs(p.x)-abs(p.y));
    if n.z<0.{n=vec3((vec2(1.)-abs(n.yx))*select(vec2(-1.),vec2(1.),n.xy>=vec2(0.)),n.z);}
    return normalize(n);
}

fn textureRandom()->f32 {var state=textureRng;let value=rngNextFloat(&state);textureRng=state;return value;}

// Non-temporal sparse bilateral experiment: four alternating ring neighbors + center.
fn sparseBilateral(x:u32,y:u32)->vec3<f32>{
    let center=averagePixel(x,y);let sigma=max(samplingParams.denoisingSigmaColor,0.0001);
    let diagonal=(jenkinsHash(x+y*frameData.x)&1u)!=0u;
    var sum=center;var weights=1.;
    for(var i=0u;i<4u;i+=1u){
        var offset=vec2<i32>(0);
        if diagonal {offset=vec2(select(-1,1,(i&1u)!=0u),select(-1,1,(i&2u)!=0u));}
        else {let sign=select(-1,1,(i&1u)!=0u);offset=select(vec2(sign,0),vec2(0,sign),i>=2u);}
        let p=vec2<i32>(i32(x),i32(y))+offset;
        if any(p<vec2(0)) || any(p>=vec2<i32>(frameData.xy)){continue;}
        let color=averagePixel(u32(p.x),u32(p.y));let delta=color-center;
        let weight=exp(-dot(delta,delta)/(2.*sigma*sigma));sum+=weight*color;weights+=weight;
    }
    return sum/weights;
}

// Pixel position of a world point in the previous frame (pinhole camera model).
fn previousPixel(p: vec3<f32>) -> vec2<f32> {
    let w = prevCamera.lowerLeftCorner - prevCamera.eye;
    let n = cross(prevCamera.horizontal, prevCamera.vertical);
    let d = p - prevCamera.eye;
    let dn = dot(d, n);
    if dn * dot(w, n) <= 0. { return vec2(-1e9); }
    let q = d * (dot(w, n) / dn) - w;
    let s = dot(q, prevCamera.horizontal) / dot(prevCamera.horizontal, prevCamera.horizontal);
    let t = dot(q, prevCamera.vertical) / dot(prevCamera.vertical, prevCamera.vertical);
    return vec2(s * f32(frameData.x), (1. - t) * f32(frameData.y));
}

// Guides for the primary vertex. `previous` is where the hit point was last frame.
fn writeGbuffer(albedo: vec3<f32>, distance: f32, normal: vec3<f32>, roughness: f32, metallic: f32, previous: vec3<f32>, kind: f32) {
    let motion = previousPixel(previous) - gbufferSample;
    let base = gbufferPixel * 3u;
    gbuffer[base] = vec4(albedo, distance);
    gbuffer[base + 1u] = vec4(normal, roughness);
    gbuffer[base + 2u] = vec4(motion, metallic, kind);
}

fn writeGbufferHit(ray: Ray, hit: Intersection) {
    let distance = hit.t * length(ray.direction);
    if TRACE_MESHES && hit.sphereIdx == 0xffffffffu {
        let m = meshMaterialOffset(hit.materialIdx);
        let uv0 = vec2(hit.u, hit.v);
        let base = (meshVec4(m) * meshTexture(m + 16u, uv0, hit.uv1, true)).rgb;
        let mr = meshTexture(m + 24u, uv0, hit.uv1, false);
        writeGbuffer(base, distance, hit.n, clamp(meshFloat(m + 8u) * mr.g, 0.045, 1.),
            clamp(meshFloat(m + 7u) * mr.b, 0., 1.), meshPrevious, 1.);
        return;
    }
    if TRACE_SPHERES {
        let material = materials[hit.materialIdx];
        var albedo = vec3(1.);
        var roughness = 1.;
        var metallic = 0.;
        switch material.id {
            case 0u, 4u: { albedo = textureLookup(material.desc1, hit.u, hit.v); }
            case 1u: { albedo = textureLookup(material.desc1, hit.u, hit.v); roughness = material.x; metallic = 1.; }
            case 2u: { roughness = 0.; }
            case 3u: {
                let sines = sin(5. * hit.p.x) * sin(5. * hit.p.y) * sin(5. * hit.p.z);
                albedo = textureLookup(material.desc2, hit.u, hit.v);
                if sines < 0. { albedo = textureLookup(material.desc1, hit.u, hit.v); }
            }
            default: {}
        }
        let moved = spheres[hit.sphereIdx].centerAndPad.xyz - spherePrevCenters[hit.sphereIdx].xyz;
        writeGbuffer(albedo, distance, hit.n, roughness, metallic, hit.p - moved, 2.);
    }
}

fn writeGbufferMiss(ray: Ray) {
    // A direction: compare against the previous camera at a very large distance.
    let previous = prevCamera.eye + normalize(ray.direction) * 1e6;
    writeGbuffer(skyRadiance(normalize(ray.direction)), 0., -normalize(ray.direction), 1., 0., previous, 0.);
}
