"""Generate WGSL inference for a trained temporal denoiser (see model.py).

    training/.venv/bin/python training/temporal/export.py RUN/best.pt OUT.wgsl

The output is appended to the renderer's shader module (`--temporal-model OUT.wgsl`).
Kernels, in dispatch order, all 8x8 workgroups over pixels:
  csTdFeatures  demodulate, reproject history/guides, write the 15 features + warp
  csTdConv1     3x3 conv 15->16 + ReLU
  csTdConv2     3x3 conv 16->16 + ReLU
  csTdBlend     head row 0: temporal blend -> x level 0
  csTdFilter0/1/2  head rows for each à-trous level; the last writes history,
                   guides for the next frame and the display radiance.
Weights are WGSL literals; every multiply-add is vectorized over 4 outputs.
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import model as M  # noqa: E402

import json  # noqa: E402
import pathlib  # noqa: E402

config = pathlib.Path(sys.argv[1]).with_name("config.json")
if config.exists():
    c = json.loads(config.read_text())
    M.configure(c["channels"], c["dilations"], c.get("half", False))
G = M.CHANNELS // 4  # vec4 groups of hidden channels
state = torch.load(sys.argv[1], map_location="cpu")
net = M.Net()
net.load_state_dict(state)
w1, b1 = net.c1.weight.detach(), net.c1.bias.detach()
w2, b2 = net.c2.weight.detach(), net.c2.bias.detach()
wh, bh = net.head.weight.detach()[:, :, 0, 0], net.head.bias.detach()


def f(x):
    return f"{float(x):.9g}"


def v4(xs):
    return "vec4(" + ",".join(f(x) for x in xs) + ")"


# Weights live in a uniform buffer (OUT.bin, vec4 entries): small loop kernels keep
# register pressure low; fully unrolled literal FMAs spilled heavily on Intel.
weights = []


def put(rows):
    """Append vec4 rows; return the starting index."""
    start = len(weights)
    for r in rows:
        r = list(r) + [0.0] * (4 - len(r))
        weights.append([float(x) for x in r])
    return start


def conv_weights(w, b, cin):
    rows = []
    for ky in range(3):
        for kx in range(3):
            for c in range(cin):
                for g in range(G):
                    rows.append(w[4 * g : 4 * g + 4, c, ky, kx].tolist())
    start = put(rows)
    bias = put([b[4 * g : 4 * g + 4].tolist() for g in range(G)])
    return start, bias


W1, B1 = conv_weights(w1, b1, M.FEATURES)
W2, B2 = conv_weights(w2, b2, M.CHANNELS)
# Head rows: G vec4 over the hidden channels, then (half-resolution models) 4 vec4
# over the 15 full-resolution features (zero-padded to 16).
HG = G + (4 if M.HALF else 0)
HW = put([
    (wh[r, 4 * g : 4 * g + 4].tolist() if g < G else
     (wh[r, M.CHANNELS + 4 * (g - G) : M.CHANNELS + 4 * (g - G) + 4].tolist()))
    for r in range(M.HEAD) for g in range(HG)
])
HB = put([bh[4 * i : 4 * i + 4].tolist() for i in range((M.HEAD + 3) // 4)])


# Inner loops are unrolled here rather than written as WGSL loops: naga guards
# every WGSL loop with an iteration counter, and on Intel those guards survived
# unrolling and cost more than the multiply-adds (conv 5.1 -> see git history).
P = 1


def conv(name, src, cin, stride, wo, bo, dst):
    body = [f"        let x{c4} = tdUnpack({src}[q + {c4}u]);" for c4 in range((cin + 3) // 4)]
    for c in range(cin):
        for g in range(G):
            body.append(f"        o{g} += tdW[base + {c * G + g}u] * x{c // 4}.{'xyzw'[c % 4]};")
    init = "\n".join(f"    var o{g} = tdW[{bo + g}u];" for g in range(G))
    store = "\n".join(f"    {dst}[i + {g}u] = tdPack(max(o{g}, vec4(0.)));" for g in range(G))
    return f"""@compute @workgroup_size(8, 8)
fn {name}(@builtin(global_invocation_id) id: vec3<u32>) {{
    let dims = tdDims();
    if id.x >= dims.x || id.y >= dims.y {{ return; }}
{init}
    for (var tap = 0u; tap < 9u; tap += 1u) {{
        let q = tdTapIn(id.xy, i32(tap % 3u) - 1, i32(tap / 3u) - 1, dims) * {stride}u;
        let base = {wo}u + tap * {cin * G}u;
{chr(10).join(body)}
    }}
    let i = (dims.x * id.y + id.x) * {G}u;
{store}
}}"""


def head_load():
    """Declarations of the head inputs at pixel i (needs `i`, `id`)."""
    lines = []
    if M.HALF:
        lines.append("    let h = (id.y / 2u) * tdDims().x + id.x / 2u;")
        lines += [f"    let hf{g} = tdUnpack(tdFeatures[i * 4u + {g}u]);" for g in range(4)]
    else:
        lines.append("    let h = i;")
    lines += [f"    let hh{g} = tdUnpack(tdH2[h * {G}u + {g}u]);" for g in range(G)]
    return "\n".join(lines)


def head_row(row):
    """Head output `row`: 1x1 conv over the hidden layer (+ full-res features)."""
    terms = [f"tdW[{HB + row // 4}u].{'xyzw'[row % 4]}"]
    terms += [f"dot(hh{g}, tdW[{HW + row * HG + g}u])" for g in range(G)]
    if M.HALF:
        terms += [f"dot(hf{g}, tdW[{HW + row * HG + G + g}u])" for g in range(4)]
    return " + ".join(terms)


def filter_kernel(level):
    d = M.DILATIONS[level]
    last = level == len(M.DILATIONS) - 1
    tail = (
        """    tdHistory[i] = tdPack(vec4(x, 0.));
    let g0 = gbuffer[i * 3u];
    let radiance = x * (g0.xyz + vec3(TD_EPS));
    tdOutput[i] = array<f32, 3>(radiance.r, radiance.g, radiance.b);"""
        if last
        else f"    tdX[{(level + 1) % 2}u * n + i] = tdPack(vec4(x, 0.));"
    )
    logits = "\n".join(f"    let l{k} = {head_row(1 + 9 * level + k)};" for k in range(9))
    m = "l0"
    for k in range(1, 9):
        m = f"max({m}, l{k})"
    exps = "\n".join(f"    let e{k} = exp(l{k} - m);" for k in range(9))
    taps = " + ".join(
        f"e{k} * tdUnpack(tdX[{level % 2}u * n + tdTap(id.xy, {(k % 3 - 1) * d}, {(k // 3 - 1) * d})]).xyz" for k in range(9))
    return f"""@compute @workgroup_size(8, 8)
fn csTdFilter{level}(@builtin(global_invocation_id) id: vec3<u32>) {{
    if id.x >= frameData.x || id.y >= frameData.y {{ return; }}
    let n = frameData.x * frameData.y;
    let i = frameData.x * id.y + id.x;
{head_load()}
{logits}
    let m = {m};
{exps}
    let x = ({taps}) / ({" + ".join(f"e{k}" for k in range(9))});
{tail}
}}"""


source = f"""// Generated by training/temporal/export.py from {sys.argv[1]}; do not edit.
const TD_EPS = {f(M.EPS)};
const TD_CHANNELS = {M.CHANNELS}u;
const TD_CONV_PIXELS = {P}u;
const TD_HALF = {1 if M.HALF else 0}u;
struct TdParams {{ historyValid: u32, parity: u32, pad1: u32, pad2: u32 }}
@group(5) @binding(0) var<storage, read_write> tdHistory: array<vec2<u32>>;
// Guides of two frames (ping-pong by tdParams.parity): distance (f32) and
// normal xyz (snorm10, zero preserved) + hit kind (2 bits).
@group(5) @binding(1) var<storage, read_write> tdPrevGuides: array<vec2<u32>>;
@group(5) @binding(2) var<storage, read_write> tdFeatures: array<vec2<u32>>;
@group(5) @binding(3) var<storage, read_write> tdH1: array<vec2<u32>>;
@group(5) @binding(4) var<storage, read_write> tdH2: array<vec2<u32>>;
@group(5) @binding(5) var<storage, read_write> tdX: array<vec2<u32>>;
@group(5) @binding(6) var<storage, read_write> tdWarp: array<vec4<u32>>;

// Intermediates are stored as four packed f16 (half the memory traffic of f32).
fn tdUnpack(v: vec2<u32>) -> vec4<f32> {{ return vec4(unpack2x16float(v.x), unpack2x16float(v.y)); }}
fn tdPack(v: vec4<f32>) -> vec2<u32> {{
    let c = clamp(v, vec4(-6e4), vec4(6e4));
    return vec2(pack2x16float(c.xy), pack2x16float(c.zw));
}}
@group(5) @binding(7) var<storage, read_write> tdOutput: array<array<f32, 3>>;
@group(5) @binding(8) var<uniform> tdParams: TdParams;
@group(5) @binding(9) var<uniform> tdW: array<vec4<f32>, {len(weights)}>;

// Pixel index of a clamped (edge-replicated) neighbour.
fn tdTap(p: vec2<u32>, dx: i32, dy: i32) -> u32 {{
    let q = clamp(vec2<i32>(p) + vec2(dx, dy), vec2(0), vec2<i32>(frameData.xy) - vec2(1));
    return u32(q.y) * frameData.x + u32(q.x);
}}

// Bilinear fetch of history (rgb) and guides at a continuous position; texels
// outside the image contribute zero (grid_sample, zeros padding).
struct TdSample {{ history: vec3<f32>, dist: f32, kind: f32, normal: vec3<f32> }}
fn tdPackGuides(dist: f32, normal: vec3<f32>, kind: f32) -> vec2<u32> {{
    let q = vec3<u32>(vec3<i32>(round(clamp(normal, vec3(-1.), vec3(1.)) * 511.))) & vec3(1023u);
    return vec2(bitcast<u32>(dist), q.x | (q.y << 10u) | (q.z << 20u) | (u32(kind + 0.5) << 30u));
}}
fn tdNormal(bits: u32) -> vec3<f32> {{
    let v = bitcast<vec3<i32>>(vec3(bits << 22u, bits << 12u, bits << 2u)) >> vec3(22u);
    return vec3<f32>(v) / 511.;
}}
fn tdBilinear(pos: vec2<f32>) -> TdSample {{
    let c = pos - vec2(0.5);
    let base = vec2<i32>(floor(c));
    let f = c - floor(c);
    var s = TdSample(vec3(0.), 0., 0., vec3(0.));
    let prev = (1u - tdParams.parity) * frameData.x * frameData.y;
    {{
        let q = base + vec2(0, 0);
        if q.x >= 0 && q.y >= 0 && q.x < i32(frameData.x) && q.y < i32(frameData.y) {{
            let w = (1. - f.x) * (1. - f.y);
            let i = u32(q.y) * frameData.x + u32(q.x);
            let g = tdPrevGuides[prev + i];
            s.history += w * tdUnpack(tdHistory[i]).xyz;
            s.dist += w * bitcast<f32>(g.x);
            s.kind += w * f32(g.y >> 30u);
            s.normal += w * tdNormal(g.y);
        }}
    }}
    {{
        let q = base + vec2(1, 0);
        if q.x >= 0 && q.y >= 0 && q.x < i32(frameData.x) && q.y < i32(frameData.y) {{
            let w = f.x * (1. - f.y);
            let i = u32(q.y) * frameData.x + u32(q.x);
            let g = tdPrevGuides[prev + i];
            s.history += w * tdUnpack(tdHistory[i]).xyz;
            s.dist += w * bitcast<f32>(g.x);
            s.kind += w * f32(g.y >> 30u);
            s.normal += w * tdNormal(g.y);
        }}
    }}
    {{
        let q = base + vec2(0, 1);
        if q.x >= 0 && q.y >= 0 && q.x < i32(frameData.x) && q.y < i32(frameData.y) {{
            let w = (1. - f.x) * f.y;
            let i = u32(q.y) * frameData.x + u32(q.x);
            let g = tdPrevGuides[prev + i];
            s.history += w * tdUnpack(tdHistory[i]).xyz;
            s.dist += w * bitcast<f32>(g.x);
            s.kind += w * f32(g.y >> 30u);
            s.normal += w * tdNormal(g.y);
        }}
    }}
    {{
        let q = base + vec2(1, 1);
        if q.x >= 0 && q.y >= 0 && q.x < i32(frameData.x) && q.y < i32(frameData.y) {{
            let w = f.x * f.y;
            let i = u32(q.y) * frameData.x + u32(q.x);
            let g = tdPrevGuides[prev + i];
            s.history += w * tdUnpack(tdHistory[i]).xyz;
            s.dist += w * bitcast<f32>(g.x);
            s.kind += w * f32(g.y >> 30u);
            s.normal += w * tdNormal(g.y);
        }}
    }}
    return s;
}}

@compute @workgroup_size(8, 8)
fn csTdFeatures(@builtin(global_invocation_id) id: vec3<u32>) {{
    if id.x >= frameData.x || id.y >= frameData.y {{ return; }}
    let i = frameData.x * id.y + id.x;
    let g0 = gbuffer[i * 3u]; let g1 = gbuffer[i * 3u + 1u]; let g2 = gbuffer[i * 3u + 2u];
    let albedo = g0.xyz; let dist = g0.w; let normal = g1.xyz; let rough = g1.w; let kind = g2.w;
    let noisy = vec3(imageBuffer[i][0], imageBuffer[i][1], imageBuffer[i][2]);
    let demod = noisy / (albedo + vec3(TD_EPS));
    var warped = vec3(0.);
    var valid = 0.;
    if tdParams.historyValid != 0u {{
        let pos = vec2<f32>(id.xy) + vec2(0.5) + g2.xy;
        let s = tdBilinear(pos);
        let inside = pos.x >= 0.5 && pos.x <= f32(frameData.x) - 0.5 && pos.y >= 0.5 && pos.y <= f32(frameData.y) - 0.5;
        let agree = abs(s.dist - dist) <= 0.1 * dist + 1e-3 && dot(s.normal, normal) >= 0.8 && abs(s.kind - kind) < 0.5;
        warped = s.history;
        valid = select(0., 1., inside && agree);
    }}
    tdWarp[i] = vec4(tdPack(vec4(warped, valid)), tdPack(vec4(demod, 0.)));
    tdPrevGuides[tdParams.parity * frameData.x * frameData.y + i] = tdPackGuides(dist, normal, kind);
    let a = log1p3(max(demod, vec3(0.)));
    let b = log1p3(max(warped, vec3(0.))) * valid;
    let al = clamp(albedo, vec3(0.), vec3(4.));
    tdFeatures[i * 4u] = tdPack(vec4(a, b.x));
    tdFeatures[i * 4u + 1u] = tdPack(vec4(b.yz, valid, al.x));
    tdFeatures[i * 4u + 2u] = tdPack(vec4(al.yz, normal.xy));
    tdFeatures[i * 4u + 3u] = tdPack(vec4(normal.z, rough, log(1. + max(dist, 0.)), 0.));
}}
fn log1p3(x: vec3<f32>) -> vec3<f32> {{ return log(vec3(1.) + x); }}

// Grid of the conv layers: full frame, or ceil(frame / 2) for half-resolution models.
fn tdDims() -> vec2<u32> {{ return select(frameData.xy, (frameData.xy + vec2(1u)) / 2u, TD_HALF != 0u); }}
fn tdTapIn(p: vec2<u32>, dx: i32, dy: i32, dims: vec2<u32>) -> u32 {{
    let q = clamp(vec2<i32>(p) + vec2(dx, dy), vec2(0), vec2<i32>(dims) - vec2(1));
    return u32(q.y) * dims.x + u32(q.x);
}}

// Half-resolution models: 2x2 mean of the features (edge rows/columns replicated),
// stored in tdH2 until conv2 overwrites it.
@compute @workgroup_size(8, 8)
fn csTdPool(@builtin(global_invocation_id) id: vec3<u32>) {{
    let dims = tdDims();
    if id.x >= dims.x || id.y >= dims.y {{ return; }}
    let q0 = min(id.xy * 2u + vec2(0u, 0u), frameData.xy - vec2(1u));
    let i0 = (q0.y * frameData.x + q0.x) * 4u;
    let q1 = min(id.xy * 2u + vec2(1u, 0u), frameData.xy - vec2(1u));
    let i1 = (q1.y * frameData.x + q1.x) * 4u;
    let q2 = min(id.xy * 2u + vec2(0u, 1u), frameData.xy - vec2(1u));
    let i2 = (q2.y * frameData.x + q2.x) * 4u;
    let q3 = min(id.xy * 2u + vec2(1u, 1u), frameData.xy - vec2(1u));
    let i3 = (q3.y * frameData.x + q3.x) * 4u;
    let o = (id.y * dims.x + id.x) * 4u;
    tdH2[o + 0u] = tdPack(0.25 * (tdUnpack(tdFeatures[i0 + 0u]) + tdUnpack(tdFeatures[i1 + 0u]) + tdUnpack(tdFeatures[i2 + 0u]) + tdUnpack(tdFeatures[i3 + 0u])));
    tdH2[o + 1u] = tdPack(0.25 * (tdUnpack(tdFeatures[i0 + 1u]) + tdUnpack(tdFeatures[i1 + 1u]) + tdUnpack(tdFeatures[i2 + 1u]) + tdUnpack(tdFeatures[i3 + 1u])));
    tdH2[o + 2u] = tdPack(0.25 * (tdUnpack(tdFeatures[i0 + 2u]) + tdUnpack(tdFeatures[i1 + 2u]) + tdUnpack(tdFeatures[i2 + 2u]) + tdUnpack(tdFeatures[i3 + 2u])));
    tdH2[o + 3u] = tdPack(0.25 * (tdUnpack(tdFeatures[i0 + 3u]) + tdUnpack(tdFeatures[i1 + 3u]) + tdUnpack(tdFeatures[i2 + 3u]) + tdUnpack(tdFeatures[i3 + 3u])));
}}

{conv("csTdConv1", "tdH2" if M.HALF else "tdFeatures", M.FEATURES, 4, W1, B1, "tdH1")}

{conv("csTdConv2", "tdH1", M.CHANNELS, G, W2, B2, "tdH2")}

@compute @workgroup_size(8, 8)
fn csTdBlend(@builtin(global_invocation_id) id: vec3<u32>) {{
    if id.x >= frameData.x || id.y >= frameData.y {{ return; }}
    let i = frameData.x * id.y + id.x;
    let packed = tdWarp[i];
    let warp = tdUnpack(packed.xy);
    let demod = tdUnpack(packed.zw).xyz;
{head_load()}
    let alpha = warp.w / (1. + exp(-({head_row(0)})));
    tdX[i] = tdPack(vec4(alpha * warp.xyz + (1. - alpha) * demod, 0.));
}}

{chr(10).join(filter_kernel(l) for l in range(len(M.DILATIONS)))}
"""
open(sys.argv[2], "w").write(source)
import numpy as np  # noqa: E402

np.asarray(weights, dtype="<f4").tofile(pathlib.Path(sys.argv[2]).with_suffix(".bin"))
print(f"wrote {sys.argv[2]} ({len(source.splitlines())} lines) and {len(weights)} weight vec4s")
