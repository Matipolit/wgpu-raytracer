# Training

Two learned stages, both trained on captures from the renderer (`--capture`) and
exported to WGSL that the renderer loads at runtime:

- **Temporal denoiser** (`training/temporal/`): runs at render resolution after the
  path tracer; the Low/Medium/High presets use `models/td-half16` and
  `models/td2-half24` (`--temporal-model FILE.wgsl`). `export.py --f16` adds the
  f16 variant the renderer picks on GPUs with shader f16.
- **Joint denoise + upscale network** (`training/upscale/`, see its README): the v3
  tiers `models/jn-*` (`--joint-model FILE.wgsl`). Too slow and unstable to use; the
  replacement is designed in
  `docs/superpowers/specs/2026-11-18-fast-denoise-upscale-network-design.md`.

Long runs on the RTX PC: `scripts/pc_run.sh` (temporal denoiser) and
`scripts/overnight_v3.sh` (captures + the joint tiers).

## Environment

```sh
python3 -m venv training/.venv
training/.venv/bin/python -m pip install -r training/requirements.lock
```

The lock pins CPU PyTorch wheels (add `--index-url https://download.pytorch.org/whl/cpu`
for those); on a CUDA machine install the matching CUDA build of the same torch
version instead (the PC uses `training-venv/`, from the cu130 index).
