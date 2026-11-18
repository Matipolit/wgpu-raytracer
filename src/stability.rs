//! `--eval-stability` (offline renders): how stable and how close to a reference the
//! image the app shows is, for any denoiser / upscaler.
//!
//! Flicker: the motion-compensated frame-to-frame change of the final image, in
//! exposure-normalized log space (log1p(k x), k = 0.18 / the first evaluated frame's
//! mean luminance), averaged over pixels whose history is valid (depth within 10%, the
//! same hit kind) and over frames after `--eval-warmup` (default 16). Motion and depth
//! come from the render-resolution G-buffer, nearest-sampled at output pixels.
//! `--eval-reference FILE` (a raw f32 rgb image at output size, e.g. `--raw-output` of
//! a many-sample render): PSNR of the last frame against it, in the same log space.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

pub struct Stability {
    output: (u32, u32),
    render: (u32, u32),
    warmup: u32,
    exposure: Option<f32>,
    previous: Option<(Vec<f32>, Vec<f32>)>,
    last: Vec<f32>,
    change: f64,
    valid: f64,
    frames: u32,
}

fn luminance(c: &[f32]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

impl Stability {
    pub fn from_args(output: (u32, u32), render: (u32, u32)) -> Option<Self> {
        crate::has_flag("--eval-stability").then(|| Self {
            output,
            render,
            warmup: crate::argument_value("--eval-warmup").map_or(16, |v| v.parse().unwrap()),
            exposure: None,
            previous: None,
            last: Vec::new(),
            change: 0.,
            valid: 0.,
            frames: 0,
        })
    }

    /// One frame: the final image (3 floats per output pixel) and the render-resolution
    /// guides (12 floats per pixel, raytracer.wgsl `gbuffer` order).
    pub fn frame(&mut self, frame: u32, image: &[f32], guides: &[f32]) {
        let (ow, oh) = self.output;
        let (rw, rh) = self.render;
        let k = *self.exposure.get_or_insert_with(|| {
            let mean = image.chunks_exact(3).map(luminance).sum::<f32>() / (ow * oh) as f32;
            0.18 / mean.max(1e-8)
        });
        let log: Vec<f32> = image.iter().map(|&v| (v.max(0.) * k).ln_1p()).collect();
        let render_pixel = |x: f32, y: f32| {
            let rx = ((x * rw as f32 / ow as f32) as u32).min(rw - 1);
            let ry = ((y * rh as f32 / oh as f32) as u32).min(rh - 1);
            (ry * rw + rx) as usize * 12
        };
        if let Some((previous, previous_guides)) = &self.previous
            && frame >= self.warmup
        {
            let (mut sum, mut count) = (0f64, 0u64);
            for y in 0..oh {
                for x in 0..ow {
                    let g = render_pixel(x as f32 + 0.5, y as f32 + 0.5);
                    let motion = [
                        guides[g + 8] * ow as f32 / rw as f32,
                        guides[g + 9] * oh as f32 / rh as f32,
                    ];
                    let (px, py) = (x as f32 + 0.5 + motion[0], y as f32 + 0.5 + motion[1]);
                    if px < 0.5 || py < 0.5 || px > ow as f32 - 0.5 || py > oh as f32 - 0.5 {
                        continue;
                    }
                    let pg = render_pixel(px, py);
                    let (depth, previous_depth) = (guides[g + 3], previous_guides[pg + 3]);
                    if (previous_depth - depth).abs() > 0.1 * depth + 1e-3
                        || (previous_guides[pg + 11] - guides[g + 11]).abs() > 0.5
                    {
                        continue;
                    }
                    // Bilinear previous frame at (px, py).
                    let (cx, cy) = (px - 0.5, py - 0.5);
                    let (bx, by) = (cx.floor() as u32, cy.floor() as u32);
                    let (fx, fy) = (cx - bx as f32, cy - by as f32);
                    let at = |x: u32, y: u32, c: usize| {
                        previous[(y.min(oh - 1) * ow + x.min(ow - 1)) as usize * 3 + c]
                    };
                    let i = (y * ow + x) as usize * 3;
                    for c in 0..3 {
                        let top = at(bx, by, c) * (1. - fx) + at(bx + 1, by, c) * fx;
                        let bottom = at(bx, by + 1, c) * (1. - fx) + at(bx + 1, by + 1, c) * fx;
                        sum += f64::from((log[i + c] - (top * (1. - fy) + bottom * fy)).abs());
                    }
                    count += 1;
                }
            }
            self.change += sum / (3 * count.max(1)) as f64;
            self.valid += count as f64 / f64::from(ow * oh);
            self.frames += 1;
        }
        self.last = image.to_vec();
        self.previous = Some((log, guides.to_vec()));
    }

    /// `--eval-dump FILE`: the last frame as raw f32 rgb (analysis).
    pub fn dump(&self) {
        if let Some(path) = crate::argument_value("--eval-dump") {
            std::fs::write(path, bytemuck::cast_slice(&self.last)).unwrap();
        }
    }

    pub fn report(&self) -> String {
        let n = f64::from(self.frames.max(1));
        let mut report = format!(
            "stability: flicker {:.4} (log units per frame, {} frames, {:.0}% of pixels compared)",
            self.change / n,
            self.frames,
            100. * self.valid / n
        );
        if let Some(path) = crate::argument_value("--eval-reference") {
            let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            let reference: Vec<f32> = bytemuck::pod_collect_to_vec(&bytes);
            assert_eq!(
                reference.len(),
                self.last.len(),
                "reference size differs from the output"
            );
            let mean = reference.chunks_exact(3).map(luminance).sum::<f32>()
                / (reference.len() / 3) as f32;
            let k = 0.18 / mean.max(1e-8);
            let mse = self
                .last
                .iter()
                .zip(&reference)
                .map(|(&a, &b)| {
                    f64::from((a.max(0.) * k).ln_1p() - (b.max(0.) * k).ln_1p()).powi(2)
                })
                .sum::<f64>()
                / reference.len() as f64;
            let shown = self.last.chunks_exact(3).map(luminance).sum::<f32>()
                / (reference.len() / 3) as f32;
            report += &format!(
                ", PSNR vs reference {:.2} dB, brightness {:.3} of the reference",
                -10. * mse.max(1e-12).log10(),
                shown / mean.max(1e-8)
            );
        }
        report
    }
}
