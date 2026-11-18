"""The reference finish (see finish.py): firefly rejection and the bounded denoise."""
import torch
import torch.nn.functional as F

LUMA = torch.tensor([0.2126, 0.7152, 0.0722])


def finish(groups, guides=None, k=None, bound=2.0, radius=5, strength=0.45, normal_width=0.02, denoise=True):
    """groups: (G, H, W, 3) group means; guides: (H, W, 6) pixel-integrated albedo and
    normal, or None (the image itself is filtered then). Returns (final, robust,
    report): the finished image, the firefly-free mean, and statistics."""
    g = groups.shape[0]
    luma_weights = LUMA.to(groups)
    mean = groups.mean(0)
    k = k or 0.18 / max(float((mean @ luma_weights).mean()), 1e-12)
    log = lambda image: torch.log1p(image.clamp(min=0) * k)
    report = {"exposure": k}
    # 1. Firefly rejection on group luminance.
    luma = groups @ luma_weights  # g, h, w
    order = luma.argsort(0)
    ranked = luma.gather(0, order)
    keep = torch.ones_like(luma, dtype=torch.bool)
    floor = 0.01 / k
    dropped = torch.zeros(luma.shape[1:], dtype=torch.bool, device=groups.device)
    still = torch.ones_like(dropped)
    for top in (g - 1, g - 2):
        rest = ranked[:top]
        limit = 2 * rest.mean(0) + 12 * rest.std(0) + floor
        still = still & (ranked[top] > limit)
        keep.scatter_(0, order[top : top + 1], ~still[None])
        dropped |= still
    count = keep.sum(0)[..., None]
    robust = (groups * keep[..., None]).sum(0) / count
    report["firefly_pixels"] = float(dropped.float().mean())
    report["firefly_energy"] = float(1 - robust.sum() / mean.sum().clamp(min=1e-20))
    # Variance of the mean, per channel, from the kept groups (3x3 smoothed: 8 groups
    # estimate it roughly).
    variance = ((groups - robust) ** 2 * keep[..., None]).sum(0) / (count - 1).clamp(min=1) / count
    variance = F.avg_pool2d(variance.permute(2, 0, 1)[None], 3, 1, 1, count_include_pad=False)[0].permute(1, 2, 0)
    error = variance.sqrt()
    report["mean_error"] = float(((k * error / (1 + k * robust)) ** 2).mean().sqrt())
    final = robust
    if denoise:
        # 2. Dual-buffer non-local means.
        half = g // 2
        halves = []
        for part in (slice(0, half), slice(half, g)):
            n = keep[part].sum(0)[..., None].clamp(min=1)
            halves.append(((groups[part] * keep[part][..., None]).sum(0) / n).permute(2, 0, 1)[None])
        var = (2 * variance).permute(2, 0, 1)[None]
        # Lighting only: divided by the pixel-integrated albedo, the image has no
        # texture left for the filter to smooth; normals keep it within a surface.
        albedo = normal = None
        if guides is not None:
            albedo = (guides[..., :3] + 0.01).permute(2, 0, 1)[None]
            normal = guides[..., 3:].permute(2, 0, 1)[None]
            halves = [x / albedo for x in halves]
            var = var / albedo**2
        r = radius
        pad = lambda t: F.pad(t, (r, r, r, r), mode="replicate")
        hh, ww = mean.shape[:2]
        outputs = []
        for guide, source in ((halves[1], halves[0]), (halves[0], halves[1])):
            pg, ps, pv = pad(guide), pad(source), pad(var)
            pn = pad(normal) if normal is not None else None
            total = torch.zeros_like(source)
            weight = torch.zeros(1, 1, hh, ww, device=groups.device)
            for dy in range(2 * r + 1):
                for dx in range(2 * r + 1):
                    q = (slice(None), slice(None), slice(dy, dy + hh), slice(dx, dx + ww))
                    distance = ((guide - pg[q]) ** 2 - (var + torch.minimum(var, pv[q]))) / (1e-10 + strength**2 * (var + pv[q]))
                    distance = F.avg_pool2d(distance.mean(1, keepdim=True), 3, 1, 1, count_include_pad=False)
                    weights = torch.exp(-distance.clamp(min=0))
                    if pn is not None:
                        weights = weights * torch.exp(-((normal - pn[q]) ** 2).sum(1, keepdim=True) / normal_width)
                    total += weights * ps[q]
                    weight += weights
            outputs.append(total / weight)
        filtered = 0.5 * (outputs[0] + outputs[1])
        if albedo is not None:
            filtered = filtered * albedo
        filtered = filtered[0].permute(1, 2, 0)
        final = torch.maximum(torch.minimum(filtered, robust + bound * error), robust - bound * error).clamp(min=0)
        report["changed_by_denoise"] = float((log(final) - log(robust)).abs().mean())
    report["brightness"] = float(final.sum() / mean.sum().clamp(min=1e-20))
    return final, robust, report
