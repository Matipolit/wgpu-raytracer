// Bloom with the dual Kawase filter (Marius Bjørge, "Bandwidth-Efficient Rendering",
// SIGGRAPH 2015): each level is downsampled with 5 bilinear taps and summed back up
// with 8, every tap one hardware-filtered texture sample. The first downsample reads
// the HDR image and Karis-averages its taps so single bright noisy pixels do not
// bloom into blobs. The display pass blends the half-resolution sum before tonemapping.

struct BloomPass {
    // Destination size (texels) and the source's texel size (1 / size).
    dstWidth: u32,
    dstHeight: u32,
    // 0: downsample from the HDR image, 1: downsample, 2: upsample (+ same-level add).
    mode: u32,
    srcWidth: u32,
    srcHeight: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> bloomPass: BloomPass;
@group(0) @binding(1) var<storage, read> bloomImage: array<f32>;
@group(0) @binding(2) var bloomSource: texture_2d<f32>;
@group(0) @binding(3) var bloomAdd: texture_2d<f32>;
@group(0) @binding(4) var bloomTarget: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var bloomSampler: sampler;

fn imageTexel(x: i32, y: i32) -> vec3<f32> {
    let cx = u32(clamp(x, 0, i32(bloomPass.srcWidth) - 1));
    let cy = u32(clamp(y, 0, i32(bloomPass.srcHeight) - 1));
    let i = (cy * bloomPass.srcWidth + cx) * 3u;
    let c = vec3(bloomImage[i], bloomImage[i + 1u], bloomImage[i + 2u]);
    // Non-finite or negative radiance never reaches the pyramid; f16 range.
    return select(vec3(0.), clamp(c, vec3(0.), vec3(60000.)), (c == c) & (abs(c) < vec3(1e30)));
}
// Bilinear from the HDR image (a buffer: four reads) at uv in [0, 1].
fn imageSample(uv: vec2<f32>) -> vec3<f32> {
    let p = uv * vec2(f32(bloomPass.srcWidth), f32(bloomPass.srcHeight)) - vec2(0.5);
    let b = vec2<i32>(floor(p));let f = p - floor(p);
    let top = mix(imageTexel(b.x, b.y), imageTexel(b.x + 1, b.y), f.x);
    let bottom = mix(imageTexel(b.x, b.y + 1), imageTexel(b.x + 1, b.y + 1), f.x);
    return mix(top, bottom, f.y);
}
fn sourceSample(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(bloomSource, bloomSampler, uv, 0.).rgb;
}
fn karis(c: vec3<f32>) -> f32 {return 1. / (1. + dot(c, vec3(0.2126, 0.7152, 0.0722)));}

@compute @workgroup_size(8, 8)
fn csBloom(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= bloomPass.dstWidth || id.y >= bloomPass.dstHeight { return; }
    let uv = (vec2<f32>(id.xy) + vec2(0.5)) / vec2(f32(bloomPass.dstWidth), f32(bloomPass.dstHeight));
    // Half a destination texel: one source texel when downsampling.
    let h = vec2(0.5) / vec2(f32(bloomPass.dstWidth), f32(bloomPass.dstHeight));
    var c = vec3(0.);
    if bloomPass.mode == 0u {
        let taps = array(imageSample(uv), imageSample(uv - h), imageSample(uv + h),
                         imageSample(uv + vec2(h.x, -h.y)), imageSample(uv - vec2(h.x, -h.y)));
        var total = 0.;
        for (var i = 0u; i < 5u; i += 1u) {
            let w = select(1., 4., i == 0u) * karis(taps[i]);
            c += w * taps[i];total += w;
        }
        c /= total;
    } else if bloomPass.mode == 1u {
        c = (4. * sourceSample(uv) + sourceSample(uv - h) + sourceSample(uv + h)
             + sourceSample(uv + vec2(h.x, -h.y)) + sourceSample(uv - vec2(h.x, -h.y))) / 8.;
    } else {
        c = (sourceSample(uv + vec2(-2. * h.x, 0.)) + sourceSample(uv + vec2(2. * h.x, 0.))
             + sourceSample(uv + vec2(0., 2. * h.y)) + sourceSample(uv + vec2(0., -2. * h.y))
             + 2. * (sourceSample(uv + vec2(-h.x, h.y)) + sourceSample(uv + h)
                     + sourceSample(uv + vec2(h.x, -h.y)) + sourceSample(uv - h))) / 12.;
        c += textureLoad(bloomAdd, vec2<i32>(id.xy), 0).rgb;
    }
    textureStore(bloomTarget, vec2<i32>(id.xy), vec4(min(c, vec3(60000.)), 1.));
}
