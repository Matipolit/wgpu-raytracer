"""Test vector for the WGSL joint network (src/raytracer/tests.rs joint_network_matches_pytorch).

    python training/upscale/parity_vector.py RUN/best.pt OUT.bin

A small random scene (13x9 output, 7x5 render, static geometry, sub-pixel motion so the
history is valid and blended) run through model.step for FRAMES frames. Little-endian:
u32 render w, h, output W, H, frames; then per frame f32 jitter (2), render radiance
(h*w*3), render guides (h*w*12), output guides (H*W*12), expected output radiance (H*W*3).
"""
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import model as M  # noqa: E402

weights, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
M.configure_from(json.loads(weights.with_name("config.json").read_text()))
net = M.Net()
net.load_state_dict(torch.load(weights, map_location="cpu"))
net.eval()
torch.manual_seed(1)
W, H, w, h, FRAMES = 13, 9, 7, 5, 3


def guides(width, height, motion):
    g = torch.zeros(1, 12, height, width)
    g[:, 0:3] = torch.rand(1, 3, height, width) * 0.8 + 0.1
    ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    g[:, 3] = 3.0 + 0.05 * xs + 0.03 * ys  # smooth distance: the history validates
    n = torch.tensor([0.2, 0.9, 0.3])
    g[:, 4:7] = (n / n.norm()).view(1, 3, 1, 1)
    g[:, 7] = torch.rand(1, height, width) * 0.8 + 0.1
    g[:, 8], g[:, 9] = motion
    g[:, 10] = (torch.rand(1, height, width) < 0.3).float()
    g[:, 11] = 1.0
    g[:, 11, 0, 0] = 0.0  # one sky pixel
    return g


hi_guides = guides(W, H, (0.3, -0.2))
lo_guides = guides(w, h, (0.3 * w / W, -0.2 * h / H))
ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
coords = torch.stack((xs + 0.5, ys + 0.5)).float().unsqueeze(0)
origin = torch.zeros(1, 2)
scale = torch.tensor([[w / W, h / H]])
history = prev = None
chunks = [np.array([w, h, W, H, FRAMES], dtype="<u4").tobytes()]
with torch.no_grad():
    for t in range(FRAMES):
        jitter = torch.rand(1, 2)
        radiance = torch.rand(1, 3, h, w) * 2.0
        low = torch.cat((radiance, lo_guides), 1)
        history, prev, result, _ = M.step(net, low, hi_guides, coords, origin, jitter, scale, history, prev, t == 0)
        chw_to_hwc = lambda x: x[0].permute(1, 2, 0).contiguous().numpy().astype("<f4").tobytes()  # noqa: E731
        chunks += [jitter.numpy().astype("<f4").tobytes(), chw_to_hwc(radiance), chw_to_hwc(lo_guides),
                   chw_to_hwc(hi_guides), chw_to_hwc(result)]
out.write_bytes(b"".join(chunks))
print(f"wrote {out}")
