# High-sample courtyard references

The application now defaults to three bounces with Russian roulette enabled.
Roulette starts before bounce four, so it does not terminate paths at that default
cap; it becomes active when the bounce limit is increased.

These reference renders deliberately keep **eight bounces** to match the existing
roulette and sky sampling comparisons. Lowering the reference to three bounces
would change the lighting target. Both courtyard views use **2048 spp, roulette
on, sky importance off, no denoising**, at 640×400 with the original camera and
exposure. Fresh frame seeds start at offset 1,000,000, separate from evaluated
captures. More samples reduce noise; 2048 spp is not a guarantee of convergence.

```sh
cargo build --release --locked
python3 performance/references/render.py
training/.venv/bin/python performance/references/update_reports.py
```

The first command creates `2048-rr/001.png`, `019.png`, float32 RGB `.f32`
files, timing `.json` files and a manifest. Completed views are reused on restart.
`render.log` records progress every 256 samples. The old training references
remain at 512 spp and the training dataset is unchanged.

The second command recalculates error metrics for the already rendered roulette
and sky experiment images using the new references, then rebuilds both visual
reports. Render timings and captured candidate images remain the original
measurements. The prior metrics are preserved as `summary-512-reference.json`.
Their source READMEs describe the original 512-spp-reference results; consult the
current generated summaries for updated error values.

Ordinary use:

```sh
cargo run --release --locked -- --sponza
# Three bounces, roulette enabled (inactive until bounce four).
cargo run --release --locked -- --sponza --bounces 8
# Eight bounces, roulette active.
```

Use `--no-russian-roulette` to disable roulette explicitly. The legacy
`--russian-roulette` flag is still accepted. Sky sampling stays opt-in.

Both experiment runners automatically select these 2048-spp references when the
reference manifest exists, otherwise falling back to the original 512-spp arrays.

Validation: unit tests, GPU compute/fragment parity, and three native-window
close checks passed. Smoke renders verified that roulette is active by default
at eight bounces and has no effect at the default three-bounce cap. Both raw
2048-spp reference arrays were checked for expected dimensions and finite values.
