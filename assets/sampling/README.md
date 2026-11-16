`blue-noise-64.bin` is original generated sampling data: two independent 64×64
void-and-cluster ranked scalar masks, packed as two unsigned normalized 16-bit
channels in little-endian u32 words. Recreate with
`training/.venv/bin/python scripts/make_blue_noise.py` (NumPy required).

The renderer's `--blue-noise-jitter` experiment shifts the spatial masks using an
R2 frame-index progression. It reads no previous rendered pixels. This is a
spatial blue-noise/progressive-jitter experiment, not a filter-trained FAST/STBN
sequence or a temporal denoiser.
