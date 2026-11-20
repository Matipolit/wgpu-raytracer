// Detail upscaler (`--detail-upscale`, upscale.rs): the denoised render-resolution image
// is demodulated by its albedo, gathered at each output pixel, remodulated by this
// frame's output-resolution albedo (raytracer.wgsl csOutputGuides), and accumulated in
// final colour with a TAA resolve, so texture detail comes from the full-resolution
// albedo and jittered edges and albedo average out over time.
//   csDetailCurrent  this frame's colour per output pixel (packed f16)
//   csDetailResolve  history reprojection, variance clipping against the 3x3
//                    neighbourhood of this frame's colour, blend 1/N (N = history
//                    length) down to alphaMin
struct DetailParams {
    renderSize: vec2<u32>,
    outputSize: vec2<u32>,
    jitter: vec2<f32>,
    historyValid: u32,
    pad0: u32,
    alphaMin: f32,
    // Variance clipping width in standard deviations.
    gamma: f32,
    // The denoiser's demodulation floor (its TD_EPS), so both demodulate alike.
    eps: f32,
    pad2: u32,
}
@group(0) @binding(0) var<uniform> params: DetailParams;
@group(0) @binding(1) var<storage, read> image: array<array<f32, 3>>;
@group(0) @binding(2) var<storage, read> lowGuides: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> outGuides: array<vec4<f32>>;
// History: rgb, then distance and history length as packed f16.
@group(0) @binding(4) var<storage, read> historyIn: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> historyOut: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> output: array<array<f32, 3>>;
@group(0) @binding(7) var<storage, read_write> current: array<vec2<u32>>;

const MAX_HISTORY = 64.;
const DETAIL_RATIO = 3.;

fn renderIndex(r: vec2<i32>) -> u32 {
    let c = clamp(r, vec2(0), vec2<i32>(params.renderSize) - vec2(1));
    return u32(c.y) * params.renderSize.x + u32(c.x);
}
fn outputIndex(p: vec2<i32>) -> u32 {
    let c = clamp(p, vec2(0), vec2<i32>(params.outputSize) - vec2(1));
    return u32(c.y) * params.outputSize.x + u32(c.x);
}
fn unpackRgb(v: vec2<u32>) -> vec3<f32> { return vec3(unpack2x16float(v.x), unpack2x16float(v.y).x); }
fn packRgb(c: vec3<f32>) -> vec2<u32> {
    let v = min(max(c, vec3(0.)), vec3(6e4));
    return vec2(pack2x16float(v.xy), pack2x16float(vec2(v.z, 0.)));
}
fn toYCoCg(c: vec3<f32>) -> vec3<f32> {
    return vec3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
fn fromYCoCg(c: vec3<f32>) -> vec3<f32> {
    return vec3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

@compute @workgroup_size(8, 8)
fn csDetailCurrent(@builtin(global_invocation_id) id: vec3<u32>) {
    let o = id.xy;
    if o.x >= params.outputSize.x || o.y >= params.outputSize.y { return; }
    // This frame's samples near the pixel centre (render pixel r's sample lies at
    // r + jitter), demodulated by the render-resolution albedo they were shaded with.
    let scale = vec2<f32>(params.renderSize) / vec2<f32>(params.outputSize);
    let q = (vec2<f32>(o) + vec2(0.5)) * scale;
    let base = vec2<i32>(floor(q - params.jitter));
    let i = o.y * params.outputSize.x + o.x;
    let g0 = outGuides[i * 3u];
    let albedo = g0.xyz;
    let depth = g0.w;
    let kind = outGuides[i * 3u + 2u].w;
    // Joint-bilateral: samples of another kind (sky vs surface) or depth barely count,
    // so demodulated values do not cross edges (a sky "albedo" holds sky radiance).
    // Directly visible emission (the guides' fourth vec4, written when a temporal
    // denoiser runs) is not albedo times light: it is gathered as it is. Demodulated
    // with the rest, a lamp's edge pixel would be scaled by the ratio of the two
    // albedos across the edge (a window frame beside a window: up to 90 times).
    let emission = params.renderSize.x * params.renderSize.y * 3u;
    var sum = vec3(0.);
    var emitted = vec3(0.);
    var weight = 0.;
    for (var dy = -1; dy <= 2; dy += 1) {
        for (var dx = -1; dx <= 2; dx += 1) {
            let r = base + vec2(dx, dy);
            let ri = renderIndex(r);
            let d = vec2<f32>(r) + params.jitter - q;
            let lo0 = lowGuides[ri * 3u];
            let relative = (lo0.w - depth) / (0.05 * depth + 1e-3);
            let same = select(1e-4, 1., abs(lowGuides[ri * 3u + 2u].w - kind) < 0.5);
            let w = exp(-2. * dot(d, d)) * same * exp(-relative * relative);
            let e = lowGuides[emission + ri].xyz;
            let c = max(vec3(image[ri][0], image[ri][1], image[ri][2]) - e, vec3(0.));
            // Light seen through glass or in a mirror is not albedo times light either,
            // and nothing marks it: the change of albedo from the sample to this pixel
            // is limited to DETAIL_RATIO either way (a dark pane beside a bright frame
            // was scaled 19 times: night Bistro 19% too bright).
            let ratio = clamp((albedo + vec3(params.eps)) / (lo0.xyz + vec3(params.eps)), vec3(1. / DETAIL_RATIO), vec3(DETAIL_RATIO));
            sum += w * c * ratio;
            emitted += w * e;
            weight += w;
        }
    }
    current[i] = packRgb((sum + emitted) / max(weight, 1e-6));
}

@compute @workgroup_size(8, 8)
fn csDetailResolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let o = vec2<i32>(id.xy);
    if id.x >= params.outputSize.x || id.y >= params.outputSize.y { return; }
    let i = id.y * params.outputSize.x + id.x;
    let c = unpackRgb(current[i]);
    // 3x3 neighbourhood of this frame's colour (YCoCg mean and deviation).
    var m1 = vec3(0.);
    var m2 = vec3(0.);
    for (var k = 0; k < 9; k += 1) {
        let n = toYCoCg(unpackRgb(current[outputIndex(o + vec2(k % 3 - 1, k / 3 - 1))]));
        m1 += n;
        m2 += n * n;
    }
    m1 /= 9.;
    let sigma = sqrt(max(m2 / 9. - m1 * m1, vec3(0.)));
    let g0 = outGuides[i * 3u];
    let g2 = outGuides[i * 3u + 2u];
    let depth = g0.w;
    var history = c;
    var length = 0.;
    if params.historyValid != 0u {
        let pos = vec2<f32>(o) + vec2(0.5) + g2.xy;
        let size = vec2<f32>(params.outputSize);
        if all(pos >= vec2(0.5)) && all(pos <= size - vec2(0.5)) {
            // Bilinear colour; distance and length from the nearest texel.
            let t = pos - vec2(0.5);
            let b = vec2<i32>(floor(t));
            let f = t - floor(t);
            let h00 = historyIn[outputIndex(b)];
            let h10 = historyIn[outputIndex(b + vec2(1, 0))];
            let h01 = historyIn[outputIndex(b + vec2(0, 1))];
            let h11 = historyIn[outputIndex(b + vec2(1, 1))];
            let rgb = mix(mix(h00.xyz, h10.xyz, f.x), mix(h01.xyz, h11.xyz, f.x), f.y);
            let nearest = unpack2x16float(bitcast<u32>(historyIn[outputIndex(vec2<i32>(floor(pos)))].w));
            if abs(nearest.x - depth) <= 0.1 * depth + 1e-3 {
                // Variance clipping: the history is pulled into this frame's colour range.
                let y = toYCoCg(rgb);
                let lo = m1 - params.gamma * sigma;
                let hi = m1 + params.gamma * sigma;
                history = fromYCoCg(clamp(y, lo, hi));
                length = nearest.y;
            }
        }
    }
    let alpha = max(1. / (length + 1.), params.alphaMin);
    let result = max(mix(history, c, alpha), vec3(0.));
    output[i] = array<f32, 3>(result.r, result.g, result.b);
    historyOut[i] = vec4(result, bitcast<f32>(pack2x16float(vec2(depth, min(length + 1., MAX_HISTORY)))));
}
