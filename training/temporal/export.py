"""Generate WGSL inference for a trained temporal denoiser (see model.py).

    training/.venv/bin/python training/temporal/export.py RUN/best.pt OUT.wgsl [--f16]

--f16 also writes OUT.f16.wgsl/.bin: the conv layers multiply and accumulate in f16
(packed math: twice the f32 rate on RDNA and Xe), used when the GPU has SHADER_F16.

The output is appended to the renderer's shader module (`--temporal-model OUT.wgsl`).
Kernels, in dispatch order, all 8x8 workgroups over pixels:
  csTdFeatures  demodulate, reproject history/guides, write the 15 features + warp
  csTdConv1     3x3 conv 15->16 + ReLU
  csTdConv2     3x3 conv 16->16 + ReLU
  csTdBlend     head row 0: temporal blend -> x level 0
  csTdFilter0/1/2  head rows for each à-trous level; the last writes history,
                   guides for the next frame and the display radiance.
Weights are WGSL literals; every multiply-add is vectorized over 4 outputs.

The model's options (config.json next to the weights; model.py) change the generated
code: `edge` adds the edge stops to the kernel logits (two guide reads per neighbour),
`length` carries the history length in the history's fourth component, `feedback`
"blend" keeps the blend instead of the filtered result, `eps` sets TD_EPS, and
`exposure` scales the radiance features by 0.18 / the mean luminance of the last
displayed frame (csTdExposure). The head's hidden layer is not supported.
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import model as M  # noqa: E402

import json  # noqa: E402
import pathlib  # noqa: E402

F16 = "--f16" in sys.argv
sys.argv = [a for a in sys.argv if a != "--f16"]
config = pathlib.Path(sys.argv[1]).with_name("config.json")
if config.exists():
    c = json.loads(config.read_text())
    M.configure(**c)
    EXPOSURE = bool(c.get("exposure", False))
else:
    EXPOSURE = False
assert M.HIDDEN == 0, "the head's hidden layer is not exported"
G = M.CHANNELS // 4  # vec4 groups of hidden channels
state = torch.load(sys.argv[1], map_location="cpu")
net = M.Net()
net.load_state_dict(state)
w1, b1 = net.c1.weight.detach(), net.c1.bias.detach()
w2, b2 = net.c2.weight.detach(), net.c2.bias.detach()
wh, bh = net.head.weight.detach()[:, :, 0, 0], net.head.bias.detach()
# Edge stops per level (depth, normal, albedo), as the model applies them.
STOPS = torch.nn.functional.softplus(net.edge.detach()).tolist() if M.EDGE else []


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
    # Even start rows: the f16 copy packs rows 2k and 2k + 1 into one slot.
    if len(weights) % 2:
        put([[0.0] * 4])
    rows = []
    for ky in range(3):
        for kx in range(3):
            for c in range(cin):
                for g in range(G):
                    rows.append(w[4 * g : 4 * g + 4, c, ky, kx].tolist())
    start = put(rows)
    if len(weights) % 2:
        put([[0.0] * 4])
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


def conv(name, src, cin, stride, wo, bo, dst, half=False):
    if half:
        return conv_f16(name, src, cin, stride, wo, bo, dst)
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


# f16 variant: the hidden layers are typed vec4<f16> (same 8-byte layout as the packed
# vec2<u32>), and an f16 copy of every weight row sits in a second uniform (binding 11)
# as pairs of rows (uniform arrays need a 16-byte stride). Conv blocks start on even
# rows and G is even, so each row's pair and half are known when generating.
def conv_f16(name, src, cin, stride, wo, bo, dst):
    assert G % 2 == 0 and wo % 2 == 0 and bo % 2 == 0
    def load(c4):
        x = f"{src}[q + {c4}u]"
        return f"vec4<f16>(tdUnpack({x}))" if src == "tdFeatures" else x
    body = [f"        let x{c4} = {load(c4)};" for c4 in range((cin + 3) // 4)]
    for c in range(cin):
        for g in range(G):
            half = "lo" if g % 2 == 0 else "hi"
            body.append(f"        o{g} = fma(tdW16[base + {(c * G + g) // 2}u].{half}, vec4(x{c // 4}.{'xyzw'[c % 4]}), o{g});")
    init = "\n".join(f"    var o{g} = tdW16[{(bo + g) // 2}u].{'lo' if g % 2 == 0 else 'hi'};" for g in range(G))
    store = "\n".join(f"    {dst}[i + {g}u] = min(max(o{g}, vec4<f16>(0.)), vec4<f16>(6e4));" for g in range(G))
    return f"""@compute @workgroup_size(8, 8)
fn {name}(@builtin(global_invocation_id) id: vec3<u32>) {{
    let dims = tdDims();
    if id.x >= dims.x || id.y >= dims.y {{ return; }}
{init}
    for (var tap = 0u; tap < 9u; tap += 1u) {{
        let q = tdTapIn(id.xy, i32(tap % 3u) - 1, i32(tap / 3u) - 1, dims) * {stride}u;
        let base = {wo // 2}u + tap * {cin * G // 2}u;
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
    # The filtered result is the history unless the blend already wrote it (csTdBlend).
    keep = "" if M.FEEDBACK == "blend" else (
        "    tdHistory[i] = tdPack(vec4(x, min(tdUnpack(tdWarp[i].zw).w + 1., TD_MAX_LENGTH)));\n" if M.LENGTH
        else "    tdHistory[i] = tdPack(vec4(x, 0.));\n")
    centre = "c0" if M.EDGE else "gbuffer[i * 3u]"
    tail = (
        keep + f"""    let radiance = x * ({centre}.xyz + vec3(TD_EPS));
    tdOutput[i] = array<f32, 3>(radiance.r, radiance.g, radiance.b);"""
        if last
        else f"    tdX[{(level + 1) % 2}u * n + i] = tdPack(vec4(x, 0.));"
    )
    if M.EDGE:
        # Edge stops: a neighbour's logit drops with its relative depth difference, its
        # normal difference and its albedo difference from the centre (model.py).
        sd, sn, sa = STOPS[level]
        rows = ["    let c0 = gbuffer[i * 3u]; let c1 = gbuffer[i * 3u + 1u].xyz;"]
        for k in range(9):
            if k == 4:
                rows.append(f"    let l{k} = {head_row(1 + 9 * level + k)};")
                continue
            rows.append(f"    let t{k} = tdTap(id.xy, {(k % 3 - 1) * d}, {(k // 3 - 1) * d}); let j{k} = t{k} * 3u;")
            rows.append(f"    let n{k} = gbuffer[j{k}];")
            rows.append(
                f"    let l{k} = {head_row(1 + 9 * level + k)} - ({f(sd)} * abs(n{k}.w - c0.w) / (c0.w + 1e-3)"
                f" + {f(sn)} * (1. - clamp(dot(gbuffer[j{k} + 1u].xyz, c1), 0., 1.))"
                f" + {f(sa / 3)} * dot(abs(n{k}.xyz - c0.xyz), vec3(1.)));")
        logits = "\n".join(rows)
    else:
        logits = "\n".join(f"    let l{k} = {head_row(1 + 9 * level + k)};" for k in range(9))
    m = "l0"
    for k in range(1, 9):
        m = f"max({m}, l{k})"
    exps = "\n".join(f"    let e{k} = exp(l{k} - m);" for k in range(9))
    taps = " + ".join(
        f"e{k} * tdUnpack(tdX[{level % 2}u * n + " + (f"t{k}" if M.EDGE and k != 4 else f"tdTap(id.xy, {(k % 3 - 1) * d}, {(k // 3 - 1) * d})") + "]).xyz"
        for k in range(9))
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


# Exposure-normalized features: 0.18 / the mean luminance of the last displayed frame
# (raytracer.wgsl csExposure; training used the reference's mean and jittered it 2x).
SCALE = (
    "    let shown = bitcast<f32>(tdHistory[frameData.x * frameData.y].x);\n"
    "    let scale = select(1., 0.18 / max(shown, 1e-6), shown > 0.);\n" if EXPOSURE
    else "    let scale = 1.;\n")
# The mean luminance of the frame just displayed, over a 64x64 grid of pixels, kept in
# the unused second half of the history buffer for the next frame's features. One
# workgroup; runs last (after the emission is added back).
EXPOSURE_KERNEL = """
var<workgroup> tdMeans: array<f32, 256>;
@compute @workgroup_size(256)
fn csTdExposure(@builtin(local_invocation_index) lane: u32) {
    let columns = min(frameData.x, 64u);
    let rows = min(frameData.y, 64u);
    var sum = 0.;
    for (var i = lane; i < columns * rows; i += 256u) {
        let x = min((i % columns) * frameData.x / columns, frameData.x - 1u);
        let y = min((i / columns) * frameData.y / rows, frameData.y - 1u);
        let c = tdOutput[frameData.x * y + x];
        sum += dot(vec3(c[0], c[1], c[2]), vec3(0.2126, 0.7152, 0.0722));
    }
    tdMeans[lane] = sum;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if lane < stride { tdMeans[lane] += tdMeans[lane + stride]; }
        workgroupBarrier();
    }
    if lane == 0u {
        tdHistory[frameData.x * frameData.y] = vec2(bitcast<u32>(tdMeans[0] / f32(columns * rows)), 0u);
    }
}
""" if EXPOSURE else ""
# The blend weight: with `length` the running mean N / (N + 1) corrected by the head.
BLEND = (
    f"""    let mean = clamp(current.w / (current.w + 1.), 1e-3, 1. - 1e-3);
    let alpha = warp.w / (1. + exp(-(log(mean / (1. - mean)) + {head_row(0)})));""" if M.LENGTH
    else f"    let alpha = warp.w / (1. + exp(-({head_row(0)})));")
KEEP_BLEND = "" if M.FEEDBACK != "blend" else (
    "    tdHistory[i] = tdPack(vec4(blend, min(current.w + 1., TD_MAX_LENGTH)));\n" if M.LENGTH
    else "    tdHistory[i] = tdPack(vec4(blend, 0.));\n")


def make_source(half):
    return f"""// Generated by training/temporal/export.py from {sys.argv[1]}; do not edit.
{F16_HEADER if half else ""}const TD_EPS = {f(M.EPS)};
const TD_CHANNELS = {M.CHANNELS}u;
const TD_CONV_PIXELS = {P}u;
const TD_HALF = {1 if M.HALF else 0}u;
const TD_MAX_LENGTH = {f(M.MAX_LENGTH)};
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
struct TdSample {{ history: vec4<f32>, dist: f32, kind: f32, normal: vec3<f32> }}
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
    var s = TdSample(vec4(0.), 0., 0., vec3(0.));
    let prev = (1u - tdParams.parity) * frameData.x * frameData.y;
    {{
        let q = base + vec2(0, 0);
        if q.x >= 0 && q.y >= 0 && q.x < i32(frameData.x) && q.y < i32(frameData.y) {{
            let w = (1. - f.x) * (1. - f.y);
            let i = u32(q.y) * frameData.x + u32(q.x);
            let g = tdPrevGuides[prev + i];
            s.history += w * tdUnpack(tdHistory[i]);
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
            s.history += w * tdUnpack(tdHistory[i]);
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
            s.history += w * tdUnpack(tdHistory[i]);
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
            s.history += w * tdUnpack(tdHistory[i]);
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
    var length = 0.;
    if tdParams.historyValid != 0u {{
        let pos = vec2<f32>(id.xy) + vec2(0.5) + g2.xy;
        let s = tdBilinear(pos);
        let inside = pos.x >= 0.5 && pos.x <= f32(frameData.x) - 0.5 && pos.y >= 0.5 && pos.y <= f32(frameData.y) - 0.5;
        let agree = abs(s.dist - dist) <= 0.1 * dist + 1e-3 && dot(s.normal, normal) >= 0.8 && abs(s.kind - kind) < 0.5;
        warped = s.history.xyz;
        valid = select(0., 1., inside && agree);
        length = s.history.w * valid;
    }}
    tdWarp[i] = vec4(tdPack(vec4(warped, valid)), tdPack(vec4(demod, length)));
    tdPrevGuides[tdParams.parity * frameData.x * frameData.y + i] = tdPackGuides(dist, normal, kind);
{SCALE}    let a = log1p3(scale * max(demod, vec3(0.)));
    let b = log1p3(scale * max(warped, vec3(0.))) * valid;
    let al = clamp(albedo, vec3(0.), vec3(4.));
    tdFeatures[i * 4u] = tdPack(vec4(a, b.x));
    tdFeatures[i * 4u + 1u] = tdPack(vec4(b.yz, valid, al.x));
    tdFeatures[i * 4u + 2u] = tdPack(vec4(al.yz, normal.xy));
    tdFeatures[i * 4u + 3u] = tdPack(vec4(normal.z, rough, log(1. + max(dist, 0.)), {'length / TD_MAX_LENGTH' if M.LENGTH else '0.'}));
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

{conv("csTdConv1", "tdH2" if M.HALF else "tdFeatures", M.FEATURES, 4, W1, B1, "tdH1", half)}

{conv("csTdConv2", "tdH1", M.CHANNELS, G, W2, B2, "tdH2", half)}

@compute @workgroup_size(8, 8)
fn csTdBlend(@builtin(global_invocation_id) id: vec3<u32>) {{
    if id.x >= frameData.x || id.y >= frameData.y {{ return; }}
    let i = frameData.x * id.y + id.x;
    let packed = tdWarp[i];
    let warp = tdUnpack(packed.xy);
    let current = tdUnpack(packed.zw);
    let demod = current.xyz;
{head_load()}
{BLEND}
    let blend = alpha * warp.xyz + (1. - alpha) * demod;
    tdX[i] = tdPack(vec4(blend, 0.));
{KEEP_BLEND}}}

{chr(10).join(filter_kernel(l) for l in range(len(M.DILATIONS)))}
{EXPOSURE_KERNEL}"""


# `enable f16;` must open the shader module: the renderer moves this line there.
# TD_W32: f32 weight rows at the start of the .bin; the f16 row pairs follow.
F16_HEADER = f"""enable f16;
const TD_W32 = {len(weights)}u;
struct TdPair {{ lo: vec4<f16>, hi: vec4<f16> }}
@group(5) @binding(11) var<uniform> tdW16: array<TdPair, {(len(weights) + 1) // 2}>;
fn tdPackH(v: vec4<f32>) -> vec4<f16> {{ return vec4<f16>(clamp(v, vec4(-6e4), vec4(6e4))); }}
"""


def make_f16_source():
    src = make_source(True)
    for name in ("tdH1", "tdH2"):
        decl = f"var<storage, read_write> {name}: array<vec2<u32>>;"
        assert src.count(decl) == 1
        src = src.replace(decl, f"var<storage, read_write> {name}: array<vec4<f16>>;")
    src = src.replace("] = tdPack(0.25 *", "] = tdPackH(0.25 *")  # pool -> tdH2
    src = src.replace("tdUnpack(tdH2[", "vec4<f32>(tdH2[")  # head inputs
    return src


import numpy as np  # noqa: E402

out = pathlib.Path(sys.argv[2])
source = make_source(False)
out.write_text(source)
np.asarray(weights, dtype="<f4").tofile(out.with_suffix(".bin"))
print(f"wrote {out} ({len(source.splitlines())} lines) and {len(weights)} weight vec4s")
if F16:
    half_out = out.with_suffix(".f16.wgsl")
    half_out.write_text(make_f16_source())
    rows = np.asarray(weights, dtype="<f4")
    if len(rows) % 2:
        rows = np.concatenate([rows, np.zeros((1, 4), "<f4")])
    packed = rows.astype("<f2").view("<f4")  # two f16 rows = one vec4 slot
    np.concatenate([np.asarray(weights, dtype="<f4").ravel(), packed.ravel()]).astype("<f4").tofile(half_out.with_suffix(".bin"))
    print(f"wrote {half_out}")
