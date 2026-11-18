"""Generate WGSL inference for a trained joint denoise + upscale network (model.py).

    python training/upscale/export.py RUN/best.pt OUT.wgsl

Writes OUT.wgsl (a self-contained module, `--joint-model OUT.wgsl`) and OUT.bin (the
weights, little-endian f32 vec4 rows, bound as a storage buffer: the larger tiers do not
fit the 64 KiB uniform limit). Kernels, in dispatch order (src/raytracer/upscale.rs):
  csJnFeatures      output res: gather this frame's samples, warp and validate the
                    history, write the 18 features, the warp record and (8x8 tiles in
                    workgroup memory) their 2x2 mean at half resolution
  csJnConv1..L      half res: 3x3 convs + ReLU, ping-pong between jnA and jnB
  csJnBlend         output res: head row 0, temporal blend
  csJnFilter0..2    output res: head rows of each a-trous level; the last remodulates
                    by the output albedo, writes the image and the next history
Every multiply-add is vectorized over 4 output channels.
"""
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import model as M  # noqa: E402

F16 = "--f16" in sys.argv
sys.argv = [a for a in sys.argv if a != "--f16"]
weights_path, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
config = weights_path.with_name("config.json")
if config.exists():
    M.configure_from(json.loads(config.read_text()))
assert M.HALF, "only half-resolution conv tiers are exported"
assert M.CHANNELS % 4 == 0
net = M.Net()
net.load_state_dict(torch.load(weights_path, map_location="cpu"))
C, L, G = M.CHANNELS, M.LAYERS, M.CHANNELS // 4
FG = (M.FEATURES + 3) // 4  # feature vec4 groups (18 -> 5, zero padded)
HG = max(G, FG)  # vec4 groups per half-resolution pixel in jnA / jnB

rows = []


def put(vectors):
    start = len(rows)
    for v in vectors:
        v = [float(x) for x in v] + [0.0] * (4 - len(v))
        rows.append(v)
    return start


def f(x):
    return f"{float(x):.9g}"


convs = []
for layer, conv in enumerate(net.convs):
    w, b = conv.weight.detach(), conv.bias.detach()
    cin = w.shape[1]
    start = put(w[4 * g : 4 * g + 4, c, ky, kx].tolist()
                for ky in range(3) for kx in range(3) for c in range(cin) for g in range(G))
    bias = put(b[4 * g : 4 * g + 4].tolist() for g in range(G))
    convs.append((cin, start, bias))
wh, bh = net.head.weight.detach()[:, :, 0, 0], net.head.bias.detach()
HEAD_ROWS = G + FG
padded = torch.cat((wh, torch.zeros(wh.shape[0], C + 4 * FG - wh.shape[1])), 1)
HW = put(padded[r, 4 * g : 4 * g + 4].tolist() for r in range(M.HEAD) for g in range(HEAD_ROWS))
HB = put(bh[4 * i : 4 * i + 4].tolist() for i in range((M.HEAD + 3) // 4))


def conv_kernel(index, cin, start, bias, src, dst, half):
    stride = HG
    if half:
        # f16 math: typed vec4<f16> layers and weights (jnW16 mirrors jnW's rows).
        body = [f"        let x{c4} = {src}[q + {c4}u];" for c4 in range((cin + 3) // 4)]
        for c in range(cin):
            for g in range(G):
                body.append(f"        o{g} = fma(jnW16[base + {c * G + g}u], vec4(x{c // 4}.{'xyzw'[c % 4]}), o{g});")
        init = "\n".join(f"    var o{g} = jnW16[{bias + g}u];" for g in range(G))
        store = "\n".join(
            f"    {dst}[i + {g}u] = min(max(o{g}, vec4<f16>(0.)), vec4<f16>(6e4));" for g in range(G))
    else:
        body = [f"        let x{c4} = jnLoadH({src}[q + {c4}u]);" for c4 in range((cin + 3) // 4)]
        for c in range(cin):
            for g in range(G):
                body.append(f"        o{g} += jnW[base + {c * G + g}u] * x{c // 4}.{'xyzw'[c % 4]};")
        init = "\n".join(f"    var o{g} = jnW[{bias + g}u];" for g in range(G))
        store = "\n".join(f"    {dst}[i + {g}u] = jnStoreH(max(o{g}, vec4(0.)));" for g in range(G))
    return f"""@compute @workgroup_size(8, 8)
fn csJnConv{index}(@builtin(global_invocation_id) id: vec3<u32>) {{
    let dims = jnHalf();
    if id.x >= dims.x || id.y >= dims.y {{ return; }}
{init}
    for (var tap = 0u; tap < 9u; tap += 1u) {{
        let p = clamp(vec2<i32>(id.xy) + vec2(i32(tap % 3u) - 1, i32(tap / 3u) - 1), vec2(0), vec2<i32>(dims) - vec2(1));
        let q = (u32(p.y) * dims.x + u32(p.x)) * {stride}u;
        let base = {start}u + tap * {cin * G}u;
{chr(10).join(body)}
    }}
    let i = (id.y * dims.x + id.x) * {stride}u;
{store}
}}"""


# The last conv writes jnB when L is odd (conv 1 reads jnA, the pooled features).
FINAL = "jnB" if L % 2 == 1 else "jnA"


def head_load():
    lines = ["    let h = (o.y / 2u) * jnHalf().x + o.x / 2u;"]
    lines += [f"    let hh{g} = jnLoadH({FINAL}[h * {HG}u + {g}u]);" for g in range(G)]
    lines += [f"    let hf{g} = jnUnpack(jnFeatures[i * {FG}u + {g}u]);" for g in range(FG)]
    return "\n".join(lines)


def head_row(r):
    terms = [f"jnW[{HB + r // 4}u].{'xyzw'[r % 4]}"]
    terms += [f"dot(hh{g}, jnW[{HW + r * HEAD_ROWS + g}u])" for g in range(G)]
    terms += [f"dot(hf{g}, jnW[{HW + r * HEAD_ROWS + G + g}u])" for g in range(FG)]
    return " + ".join(terms)


def filter_kernel(level):
    d = M.DILATIONS[level]
    last = level == len(M.DILATIONS) - 1
    logits = "\n".join(f"    let l{k} = {head_row(1 + 9 * level + k)};" for k in range(9))
    m = "l0"
    for k in range(1, 9):
        m = f"max({m}, l{k})"
    exps = "\n".join(f"    let e{k} = exp(l{k} - m);" for k in range(9))
    taps = " + ".join(
        f"e{k} * jnUnpack(jnX[{level % 2}u * n + jnTap(o, {(k % 3 - 1) * d}, {(k // 3 - 1) * d})]).xyz" for k in range(9))
    if last:
        # csJnFeatures already wrote the history's guide half (.zw).
        tail = """    let g0 = outGuides[i * 3u];
    let radiance = x * (g0.xyz + vec3(JN_EPS));
    output[i] = array<f32, 3>(radiance.r, radiance.g, radiance.b);
    let rgbd = jnPack(vec4(x, g0.w));
    historyOut[i].x = rgbd.x;
    historyOut[i].y = rgbd.y;"""
    else:
        tail = f"    jnX[{(level + 1) % 2}u * n + i] = jnPack(vec4(x, 0.));"
    return f"""@compute @workgroup_size(8, 8)
fn csJnFilter{level}(@builtin(global_invocation_id) id: vec3<u32>) {{
    let o = id.xy;
    if o.x >= jn.outputSize.x || o.y >= jn.outputSize.y {{ return; }}
    let n = jn.outputSize.x * jn.outputSize.y;
    let i = o.y * jn.outputSize.x + o.x;
{head_load()}
{logits}
    let m = {m};
{exps}
    let x = ({taps}) / ({" + ".join(f"e{k}" for k in range(9))});
{tail}
}}"""


ping = ["jnA", "jnB"]


def conv_source(half):
    return "\n\n".join(conv_kernel(k + 1, cin, start, bias, ping[k % 2], ping[(k + 1) % 2], half)
                       for k, (cin, start, bias) in enumerate(convs))


def precision(half):
    """Declarations that differ between the f32 and the f16 (`enable f16`) variants."""
    if half:
        return f"""// f16 variant: hidden layers typed vec4<f16>, f16 copies of the weight rows.
const JN_W32 = {len(rows)}u;
@group(0) @binding(8) var<storage, read_write> jnA: array<vec4<f16>>;
@group(0) @binding(9) var<storage, read_write> jnB: array<vec4<f16>>;
@group(0) @binding(13) var<storage, read> jnW16: array<vec4<f16>>;
fn jnLoadH(v: vec4<f16>) -> vec4<f32> {{ return vec4<f32>(v); }}
fn jnStoreH(v: vec4<f32>) -> vec4<f16> {{ return vec4<f16>(clamp(v, vec4(-6e4), vec4(6e4))); }}"""
    return """@group(0) @binding(8) var<storage, read_write> jnA: array<vec2<u32>>;
@group(0) @binding(9) var<storage, read_write> jnB: array<vec2<u32>>;
fn jnLoadH(v: vec2<u32>) -> vec4<f32> { return jnUnpack(v); }
fn jnStoreH(v: vec4<f32>) -> vec2<u32> { return jnPack(v); }"""

def make_source(half):
    return f"""{"enable f16;" + chr(10) if half else ""}// Generated by training/upscale/export.py from {weights_path}; do not edit.
const JN_EPS = {f(M.EPS)};
const JN_CHANNELS = {C}u;
const JN_LAYERS = {L}u;
const JN_FEATURE_GROUPS = {FG}u;
const JN_HIDDEN_GROUPS = {HG}u;
struct JnParams {{ renderSize: vec2<u32>, outputSize: vec2<u32>, jitter: vec2<f32>, historyValid: u32, pad: u32 }}
@group(0) @binding(0) var<uniform> jn: JnParams;
// This frame at render resolution: 1-spp radiance and the tracer's guides (3 vec4 per
// pixel: albedo + distance, normal + roughness, motion + metallic + kind).
@group(0) @binding(1) var<storage, read> image: array<array<f32, 3>>;
@group(0) @binding(2) var<storage, read> lowGuides: array<vec4<f32>>;
// The same guides at output resolution (raytracer.wgsl csOutputGuides).
@group(0) @binding(3) var<storage, read> outGuides: array<vec4<f32>>;
// History (ping-pong), packed f16: demodulated rgb + distance (.xy), normal + kind (.zw).
@group(0) @binding(4) var<storage, read> historyIn: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> historyOut: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read_write> output: array<array<f32, 3>>;
// Intermediates as packed f16 vec4s: features (output res), hidden layers (half res),
// the a-trous ping-pong and the warp record (warped rgb + validity, gathered rgb).
@group(0) @binding(7) var<storage, read_write> jnFeatures: array<vec2<u32>>;
@group(0) @binding(10) var<storage, read_write> jnX: array<vec2<u32>>;
@group(0) @binding(11) var<storage, read_write> jnWarp: array<vec4<u32>>;
@group(0) @binding(12) var<storage, read> jnW: array<vec4<f32>>;

fn jnUnpack(v: vec2<u32>) -> vec4<f32> {{ return vec4(unpack2x16float(v.x), unpack2x16float(v.y)); }}
fn jnPack(v: vec4<f32>) -> vec2<u32> {{
    let c = clamp(v, vec4(-6e4), vec4(6e4));
    return vec2(pack2x16float(c.xy), pack2x16float(c.zw));
}}
{precision(half)}
fn jnHalf() -> vec2<u32> {{ return (jn.outputSize + vec2(1u)) / 2u; }}
// Output pixel index of a neighbour, edges replicated.
fn jnTap(p: vec2<u32>, dx: i32, dy: i32) -> u32 {{
    let q = clamp(vec2<i32>(p) + vec2(dx, dy), vec2(0), vec2<i32>(jn.outputSize) - vec2(1));
    return u32(q.y) * jn.outputSize.x + u32(q.x);
}}
fn jnRenderIndex(r: vec2<i32>) -> u32 {{
    let c = clamp(r, vec2(0), vec2<i32>(jn.renderSize) - vec2(1));
    return u32(c.y) * jn.renderSize.x + u32(c.x);
}}

// The features of an 8x8 tile, for the 2x2 pool (half-resolution conv input).
var<workgroup> jnTile: array<array<vec4<f32>, {FG}>, 64>;
@compute @workgroup_size(8, 8)
fn csJnFeatures(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    // Threads past the edge compute the edge pixel's features (the pool replicates the
    // last row / column of odd sizes) and write nothing per pixel.
    let owned = id.x < jn.outputSize.x && id.y < jn.outputSize.y;
    let o = min(id.xy, jn.outputSize - vec2(1u));
    let i = o.y * jn.outputSize.x + o.x;
    // This frame's samples near the pixel centre (render pixel r's sample lies at
    // r + jitter), demodulated by the render-resolution albedo.
    let scale = vec2<f32>(jn.renderSize) / vec2<f32>(jn.outputSize);
    let q = (vec2<f32>(o) + vec2(0.5)) * scale;
    let base = vec2<i32>(floor(q - jn.jitter));
    var sum = vec3(0.);
    var weight = 0.;
    var nearest = 1e9;
    for (var dy = -1; dy <= 2; dy += 1) {{
        for (var dx = -1; dx <= 2; dx += 1) {{
            let r = base + vec2(dx, dy);
            let ri = jnRenderIndex(r);
            let d = vec2<f32>(r) + jn.jitter - q;
            let d2 = dot(d, d);
            let c = vec3(image[ri][0], image[ri][1], image[ri][2]);
            let w = exp(-2. * d2);
            sum += w * c / (lowGuides[ri * 3u].xyz + vec3(JN_EPS));
            weight += w;
            nearest = min(nearest, d2);
        }}
    }}
    let current = sum / max(weight, 1e-6);
    let g0 = outGuides[i * 3u];
    let g1 = outGuides[i * 3u + 1u];
    let g2 = outGuides[i * 3u + 2u];
    let dist = g0.w;
    let normal = g1.xyz;
    let kind = g2.w;
    // History at the previous position (bilinear, texels outside count 0).
    var warped = vec3(0.);
    var valid = 0.;
    if jn.historyValid != 0u {{
        let pos = vec2<f32>(o) + vec2(0.5) + g2.xy;
        let size = vec2<f32>(jn.outputSize);
        let inside = all(pos >= vec2(0.5)) && all(pos <= size - vec2(0.5));
        let c = pos - vec2(0.5);
        let b = vec2<i32>(floor(c));
        let t = c - floor(c);
        var s0 = vec4(0.);
        var s1 = vec4(0.);
        for (var k = 0; k < 4; k += 1) {{
            let e = vec2(k & 1, k >> 1);
            let p = b + e;
            if p.x < 0 || p.y < 0 || p.x >= i32(jn.outputSize.x) || p.y >= i32(jn.outputSize.y) {{ continue; }}
            let w = select(1. - t.x, t.x, e.x == 1) * select(1. - t.y, t.y, e.y == 1);
            let packed = historyIn[u32(p.y) * jn.outputSize.x + u32(p.x)];
            s0 += w * jnUnpack(packed.xy);
            s1 += w * jnUnpack(packed.zw);
        }}
        warped = s0.xyz;
        let agree = abs(s0.w - dist) <= 0.1 * dist + 1e-3 && dot(s1.xyz, normal) >= 0.8 && abs(s1.w - kind) < 0.5;
        valid = select(0., 1., inside && agree);
    }}
    let a = log(vec3(1.) + max(current, vec3(0.)));
    let h = log(vec3(1.) + max(warped, vec3(0.))) * valid;
    let al = clamp(g0.xyz, vec3(0.), vec3(4.));
    let inverseScale = 2. / (scale.x + scale.y);
    let features = array(
        vec4(a, min(weight, 4.)),
        vec4(min(sqrt(nearest), 2.), h),
        vec4(valid, al),
        vec4(normal, g1.w),
        vec4(log(1. + max(dist, 0.)), inverseScale, 0., 0.),
    );
    if owned {{
        jnWarp[i] = vec4(jnPack(vec4(warped, valid)), jnPack(vec4(current, 0.)));
        // This frame's history guides (the last filter level adds rgb + distance).
        let guides = jnPack(vec4(normal, kind));
        historyOut[i].z = guides.x;
        historyOut[i].w = guides.y;
        for (var g = 0u; g < {FG}u; g += 1u) {{ jnFeatures[i * {FG}u + g] = jnPack(features[g]); }}
    }}
    // 2x2 mean into the half-resolution grid (8x8 tiles start on even pixels).
    jnTile[lid.y * 8u + lid.x] = features;
    workgroupBarrier();
    let halfPixel = id.xy / 2u;
    if (lid.x & 1u) == 0u && (lid.y & 1u) == 0u && all(halfPixel < jnHalf()) {{
        let t = lid.y * 8u + lid.x;
        let out = (halfPixel.y * jnHalf().x + halfPixel.x) * {HG}u;
        for (var g = 0u; g < {FG}u; g += 1u) {{
            jnA[out + g] = jnStoreH(0.25 * (jnTile[t][g] + jnTile[t + 1u][g] + jnTile[t + 8u][g] + jnTile[t + 9u][g]));
        }}
    }}
}}

{conv_source(half)}

@compute @workgroup_size(8, 8)
fn csJnBlend(@builtin(global_invocation_id) id: vec3<u32>) {{
    let o = id.xy;
    if o.x >= jn.outputSize.x || o.y >= jn.outputSize.y {{ return; }}
    let i = o.y * jn.outputSize.x + o.x;
    let packed = jnWarp[i];
    let warp = jnUnpack(packed.xy);
    let current = jnUnpack(packed.zw).xyz;
{head_load()}
    let alpha = warp.w / (1. + exp(-({head_row(0)})));
    jnX[i] = jnPack(vec4(alpha * warp.xyz + (1. - alpha) * current, 0.));
}}

{chr(10).join(filter_kernel(level) for level in range(len(M.DILATIONS)))}
"""
source = make_source(False)
out.write_text(source)
np.asarray(rows, dtype="<f4").tofile(out.with_suffix(".bin"))
print(f"wrote {out} ({len(source.splitlines())} lines, {L} convs of {C} channels) and {len(rows)} weight vec4s")
if F16:
    # The f16 variant (used with SHADER_F16): its .bin is the f32 rows, then the same
    # rows as f16 (JN_W32 says where they start).
    half_out = out.with_suffix(".f16.wgsl")
    half_out.write_text(make_source(True))
    f32 = np.asarray(rows, dtype="<f4")
    with open(half_out.with_suffix(".bin"), "wb") as f:
        f.write(f32.tobytes())
        f.write(f32.astype("<f2").tobytes())
    print(f"wrote {half_out}")
