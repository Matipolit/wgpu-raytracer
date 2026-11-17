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
override GGX_VNDF: bool = false;
override GGX_BOUNDED:bool=false;

override SPARSE_DENOISE:bool=false;
// Diagnostics: output (BVH nodes visited, triangles tested, rays traced) per pixel instead of radiance.
override TRAVERSAL_STATS:bool=false;
// Write primary-hit guides and motion vectors (temporal denoiser input).
override GBUFFER:bool=false;
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
override OCT_BVH:bool=false;
// Hardware ray queries (--hw-rt); see hwrt.wgsl.
override HW_RT:bool=false;
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
fn csAccumulate(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    quadLane = lane;
    let x = id.x + (samplingParams.traceOrigin & 0xffffu);
    let y = id.y + (samplingParams.traceOrigin >> 16u);
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
    return vec4(uncharted2(vec3(color[0], color[1], color[2])), 1f);
}

fn uncharted2(x: vec3<f32>) -> vec3<f32> {
    // Based on uncharted2 tonemapping function
    // https://dmnsgn.github.io/glsl-tone-map/
    // Automatic exposure maps the log-average luminance to middle grey (0.18);
    // the user exposure then acts as compensation on top of it.
    let automatic = select(1f, 0.18 / exp(displayExposure[0]), displayExposure[1] != 0f);
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
    for (var i = 0u; i < numSamples; i += 1u) {
        if BLUE_NOISE_PATH {
            blueNoiseDim = 0u;
            blueNoisePixel = vec2(x, y);
            blueNoiseShift = fract(0.6180339887 * f32((frameData.z * numSamples + i) % 65536u));
        }
        var jitter=vec2(rngNextFloat(rngState),rngNextFloat(rngState));
        if CENTER_JITTER {jitter=vec2(0.5);}
        if VIS_PRIMARY {jitter=visibilityJitter();}
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
        color += rayColor(primaryRay, rngState);
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

        if intersection(ray, &hit) {
            if GBUFFER && bounce == 0u {
                let shaded = TRACE_MESHES && hit.sphereIdx == 0xffffffffu && !(SKIP_FINAL_SCATTER && samplingParams.numBounces == 1u);
                if shaded { gbufferPending = true; } else { writeGbufferHit(ray, hit); }
            }
            if TRACE_MESHES && hit.sphereIdx == 0xffffffffu {
                let m = meshMaterialOffset(hit.materialIdx);
                color += throughput * meshVec3(m+4u) * meshTexture(m+32u,vec2(hit.u,hit.v),hit.uv1,true).rgb;
                if SKIP_FINAL_SCATTER && bounce+1u==samplingParams.numBounces {
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
                            color += throughput * skyRadiance(direction) * eval.rgb * misWeight(pdf,eval.a)/pdf;
                        }
                    }
                    let sampled = samplePbr(surface,hit,rngState);
                    scatter = sampled.scatter;
                    skyMisWeight = misWeight(sampled.pdf,skyPdf(scatter.ray.direction));
                } else {scatter = scatterPbr(ray,hit,rngState);}
                if GBUFFER && gbufferPending {
                    gbufferPending = false;
                    writeGbuffer(shadedBase, hit.t * length(ray.direction), hit.n, shadedRough, shadedMetal, hit.p, 1.);
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
                    break;
                }

                skyMisWeight = 1.;packedValid=false;if OPACITY_APPROX {pathDiffuse=false;}
                var scatter = scatterRay(ray, hit, material, rngState);
                ray = scatter.ray;
                throughput *= scatter.throughput;
                if max(throughput.r,max(throughput.g,throughput.b)) <= 0. {break;}
            }
        } else {
            if GBUFFER && bounce == 0u { writeGbufferMiss(ray); }
            color += throughput * skyRadiance(normalize(ray.direction)) * skyMisWeight;

            break;
        }
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
    sunDirection: vec3<f32>,
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

fn skyRadiance(v: vec3<f32>) -> vec3<f32> {
    let theta=acos(clamp(v.y,-1.,1.));
    let gamma=acos(clamp(dot(v,skyState.sunDirection),-1.,1.));
    return vec3(radiance(theta,gamma,0u),radiance(theta,gamma,1u),radiance(theta,gamma,2u));
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
fn skyOccluded(ray: Ray) -> bool {
    let diffuse=pathDiffuse;if OPACITY_APPROX {pathDiffuse=false;}
    let blocked=skyOccludedExact(ray);
    if OPACITY_APPROX {pathDiffuse=diffuse;}
    return blocked;
}
fn skyOccludedExact(ray: Ray) -> bool {
    if ALPHA_CAP {alphaQueries=0u;}
    for(var i=0u;i<SPHERE_COUNT;i+=1u) {
        var hit=Intersection();
        if rayIntersectSphere(ray,i,MIN_T,MAX_T,&hit) {return true;}
    }
    return TRACE_MESHES && meshOccluded(ray);
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
            clamp(meshFloat(m + 7u) * mr.b, 0., 1.), hit.p, 1.);
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
