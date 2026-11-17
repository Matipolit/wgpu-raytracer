// Temporal upscaler (`--render-scale S`, see upscale.rs): the non-learned baseline for
// the joint denoise + upscale network. Inputs at render resolution: this frame's image
// (raw 1 spp, or the temporal denoiser's output) and the tracer's G-buffer, traced with
// one sub-pixel jitter shared by all pixels. Output and history at output resolution.
//
// Per output pixel: gather this frame's samples near the pixel centre (Gaussian in render
// pixels, their jittered positions known), reproject the history with the motion vector
// of the nearest sample, reject it on depth/kind disagreement, optionally clamp it to the
// neighbourhood's colour range (only meaningful for denoised input), and blend with a
// weight that grows with how close this frame's samples landed.
struct UpscaleParams {
    renderSize: vec2<u32>,
    outputSize: vec2<u32>,
    jitter: vec2<f32>,
    historyValid: u32,
    clampHistory: u32,
    // Blend weight of this frame: alphaMin when its samples are far, alphaMax when close.
    alphaMin: f32,
    alphaMax: f32,
    pad0: u32,
    pad1: u32,
}
@group(0) @binding(0) var<uniform> params: UpscaleParams;
@group(0) @binding(1) var<storage, read> image: array<array<f32, 3>>;
@group(0) @binding(2) var<storage, read> gbuffer: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> historyIn: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> historyOut: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> output: array<array<f32, 3>>;

fn renderIndex(r: vec2<i32>) -> u32 {
    let c = clamp(r, vec2(0), vec2<i32>(params.renderSize) - vec2(1));
    return u32(c.y) * params.renderSize.x + u32(c.x);
}

// Bilinear history (rgb, depth) at a continuous output position; texels outside count 0.
fn historyAt(pos: vec2<f32>) -> vec4<f32> {
    let c = pos - vec2(0.5);
    let base = vec2<i32>(floor(c));
    let f = c - floor(c);
    var sum = vec4(0.);
    var weight = 0.;
    for (var k = 0; k < 4; k += 1) {
        let o = vec2(k & 1, k >> 1);
        let q = base + o;
        if q.x < 0 || q.y < 0 || q.x >= i32(params.outputSize.x) || q.y >= i32(params.outputSize.y) { continue; }
        let w = select(1. - f.x, f.x, o.x == 1) * select(1. - f.y, f.y, o.y == 1);
        sum += w * historyIn[u32(q.y) * params.outputSize.x + u32(q.x)];
        weight += w;
    }
    return sum / max(weight, 1e-6);
}

@compute @workgroup_size(8, 8)
fn csUpscale(@builtin(global_invocation_id) id: vec3<u32>) {
    let o = id.xy;
    if o.x >= params.outputSize.x || o.y >= params.outputSize.y { return; }
    let scale = vec2<f32>(params.renderSize) / vec2<f32>(params.outputSize);
    // Output pixel centre in render-pixel units; render pixel r's sample lies at r + jitter.
    let q = (vec2<f32>(o) + vec2(0.5)) * scale;
    let base = vec2<i32>(floor(q - params.jitter));
    var sum = vec3(0.);
    var weightSum = 0.;
    var lo = vec3(1e30);
    var hi = vec3(-1e30);
    var nearest = 1e30;
    var nearestIndex = 0u;
    for (var dy = -1; dy <= 2; dy += 1) {
        for (var dx = -1; dx <= 2; dx += 1) {
            let r = base + vec2(dx, dy);
            let i = renderIndex(r);
            let d = vec2<f32>(r) + params.jitter - q;
            let d2 = dot(d, d);
            let c = vec3(image[i][0], image[i][1], image[i][2]);
            let w = exp(-2. * d2);
            sum += w * c;
            weightSum += w;
            if d2 < 2.25 {
                lo = min(lo, c);
                hi = max(hi, c);
            }
            if d2 < nearest {
                nearest = d2;
                nearestIndex = i;
            }
        }
    }
    let current = sum / max(weightSum, 1e-6);
    let depth = gbuffer[nearestIndex * 3u].w;
    let motion = gbuffer[nearestIndex * 3u + 2u].xy / scale;
    let previous = vec2<f32>(o) + vec2(0.5) + motion;
    let inside = all(previous >= vec2(0.)) && all(previous <= vec2<f32>(params.outputSize));
    var history = historyAt(previous);
    let valid = params.historyValid != 0u && inside && abs(history.w - depth) <= 0.1 * depth + 1e-3;
    var color = current;
    if valid {
        var h = history.rgb;
        if params.clampHistory != 0u { h = clamp(h, lo, hi); }
        // Closest sample within half a render pixel: full confidence.
        let confidence = exp(-2. * nearest);
        let alpha = mix(params.alphaMin, params.alphaMax, confidence);
        color = mix(h, current, alpha);
    }
    let i = o.y * params.outputSize.x + o.x;
    historyOut[i] = vec4(color, depth);
    output[i] = array<f32, 3>(color.r, color.g, color.b);
}
