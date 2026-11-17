# Sun disk, Bistro at night, launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A sun disk (with direct sun sampling) on by default, a Bistro night preset, and a launcher menu with graphics settings; deploy the build to the RTX PC.

**Architecture:**
- **Sun:** the sun is added to the analytic sky in WGSL (`skyRadiance = atmosphereRadiance + sunRadiance`). Its radiance, colour and size come from the CPU via `GpuSkyState`.
- **Sun sampling:** mesh PBR bounces take one shadow ray toward the sun disk, combined with BSDF hits on the disk by power-heuristic MIS.
- **Night:** a sky flag that scales the atmosphere down and turns the sun off.
- **Launcher:** a separate egui window. It builds a command line and restarts the binary with it.

**Tech Stack:** Rust, wgpu 30, WGSL, egui 0.37 / winit 0.30.

**Spec:** `docs/superpowers/specs/2026-11-17-sun-night-launcher-design.md`

## Global Constraints

- **No breaking changes:** existing command lines keep working unchanged (capture, benchmark, offline, `scripts/pc_run.sh`). The only exception is the new default sun on mesh scenes.
- **New flags:**
  - `--no-sun`, `--night`;
  - `--launcher`, `--fullscreen`;
  - `--size WxH` (initial window size in the interactive app).
- **Sun disk:** angular radius 0.2666°. At elevation 45° and turbidity 4, sun horizontal illuminance is 4× the sky's.
- **Sphere demo scene:** sun off. Legacy sphere materials keep their sampling.
- **Saved launcher settings:** `~/.config/wgpu-raytracer/launcher.txt`.
- **Quality bar:** zero warnings (`#![deny(clippy::pedantic)]` in main), `cargo fmt`, all tests pass.

## Review Focus

1. **Grazing sun, or sun below the horizon (zenith ≥ 90°).** Expect no NaNs, a red/dim sun near the horizon, and no sun below it. Owned by Task 1: `sun_state_handles_horizon_and_below`.
2. **Launcher with no models present** (fresh clone without assets, or missing denoiser files). Those scenes/models are hidden and Launch still works for the sphere demo. Owned by Task 3: `missing_models_are_hidden`.
3. **Corrupt or old launcher.txt.** Unknown or bad lines are ignored and defaults are used. Owned by Task 3: `settings_parse_ignores_garbage`.
4. **`--skip-final-scatter` / last bounce.** The sun is not double-counted or lost at the final vertex: NEE is weighted 1 when no BSDF continuation is traced. Owned by Task 2: an energy test at bounce count 1.
5. **Sky importance (`--sky-importance`) with the sun.** Sky NEE must use atmosphere-only radiance, otherwise the sun is counted twice. Owned by Task 2: the existing `csSkyEnergy` switched to `atmosphereRadiance`.

---

### Task 1: Sun disk and night sky in the sky state

**Files:**
- Modify: `src/raytracer/mod.rs`:
  - `SkyParams` (~2195): add `sun: bool` (default true) and `night: bool` (default false);
  - `GpuSkyState` (~2506);
  - `to_sky_state`;
  - `sky_sampling_table` stays atmosphere-only.
- Modify: `src/raytracer/raytracer.wgsl`: `SkyState` (~754); split `skyRadiance` (~986) into `atmosphereRadiance` + `sunRadiance`.
- Modify: `src/raytracer/tests/fragment_reference.wgsl` `SkyState`, mirroring the layout (sun unused there).
- Modify: `src/main.rs` `scene_sky`:
  - sphere scene: `sun: false`;
  - mesh scenes: `sun: !has_flag("--no-sun")`, `night: has_flag("--night")`.
- Test: `src/raytracer/tests.rs`.

**Interfaces:**
- Produces, in Rust: `GpuSkyState` with these fields:
  - `sun_direction: [f32; 4]`, where `w` = cos(angular radius);
  - `sun_radiance: [f32; 4]`, RGB, with `w` = solid angle; zero when the sun is off or below the horizon.
  - Followed by the `sampling` table.
- Produces, in WGSL:
  - `skyState.sunDirection: vec4<f32>`, `skyState.sunRadiance: vec4<f32>`;
  - `fn atmosphereRadiance(v: vec3<f32>) -> vec3<f32>`;
  - `fn sunRadiance(v: vec3<f32>) -> vec3<f32>`: the disk radiance inside the cone, else 0;
  - `fn skyRadiance(v)` = the sum of the two.
- Produces: `pub fn sun_radiance(params: &SkyParams) -> [f32; 3]`, pure, for tests.

- [ ] **Step 1: Write failing tests** in `src/raytracer/tests.rs`:
  - `sun_is_four_times_sky_at_reference`: at elevation 45° and turbidity 4, the luminance of `sun_radiance` × solid angle × sin(45°), divided by the sky's horizontal illuminance, is in 3.6..4.4. Sky illuminance is integrated over the upper hemisphere with the Rust radiance evaluation already used by `sky_sampling_table`.
  - `sun_state_handles_horizon_and_below`: at zenith 89° all components are finite and > 0, and r/b exceeds r/b at zenith 30°. At zenith 95° the result is all zero.
  - `night_sky_is_dark_and_sunless`: with `night: true`, the sun is zero and the atmosphere radiances are < 1% of day.
- [ ] **Step 2:** `cargo test --release --locked sun_ night_` → FAIL (does not compile).
- [ ] **Step 3: Implement.** Sun colour and strength: `C · T_rgb(m, turbidity)`.
  - Optical depth per channel at 680/550/440 nm: Rayleigh `τ_R = 0.008735 λµm^-4.08`; aerosol `τ_A = β λµm^-1.3` with `β = 0.04608·turbidity − 0.04586`.
  - Air mass, Kasten–Young: `m = 1/(cos z + 0.50572 (96.07995 − z°)^-1.6364)`.
  - Then `T = exp(−(τ_R+τ_A)·m)`.
  - `C` is computed in `sun_radiance` from the reference sky (45°, turbidity 4), so that the reference ratio is exactly 4.
  - Night: atmosphere radiances × `[0.0015, 0.002, 0.003]`, sun off.
  - `sky_sampling_table` stays atmosphere-only.
- [ ] **Step 4:** run the new tests plus the whole suite (`cargo test --release --locked`) → PASS. `compute_matches_fragment_reference…` uses `sun: false`.
- [ ] **Step 5:** Render Sponza and Bistro at 640x400 with 256 spp, with and without `--no-sun`, plus `--night`, and eyeball them. Commit `Add a sun disk and a night sky`.

### Task 2: Direct sun sampling with MIS for mesh surfaces

**Files:**
- Modify: `src/raytracer/raytracer.wgsl`:
  - `rayColor` (~414–495): sun NEE next to the existing `SKY_IMPORTANCE` block;
  - escaped rays apply `skyMisWeight` to the atmosphere and `sunMisWeight` to the sun;
  - sky NEE uses `atmosphereRadiance`.
- Modify: `src/raytracer/wavefront.wgsl:121`: unchanged semantics (`skyRadiance`, weight 1; the wavefront path has no NEE).
- Test: `src/raytracer/tests/mesh_cases.rs`.

**Interfaces:**
- Consumes: Task 1 WGSL functions and fields.
- Produces:
  - `fn sampleSun(rng: ptr<function,u32>) -> vec3<f32>`: a uniform direction in the sun cone;
  - `fn sunPdf(direction: vec3<f32>) -> f32`: `1/solidAngle` inside the cone, else 0;
  - override `SUN: bool`, true when the sun radiance is non-zero. It is set from `SkyParams.sun && !night` in `trace_constants`, and a change rebuilds pipelines like any override.

- [ ] **Step 1: Write the failing GPU test** `csSunEnergy` / `sun_nee_and_bsdf_hits_sum_to_the_analytic_integral`. Setup: a Lambertian surface (albedo 0.8, normal +z), sun at zenith 30°.
  - Average over 4096 samples of NEE `ev.rgb·sunRadiance·misWeight(sunPdf, ev.a)/sunPdf`, plus BSDF samples `throughput·sunRadiance·misWeight(pdf, sunPdf)`.
  - Must equal `0.8/π · cos(30°) · L_sun · Ω` within 3%.
  - Also switch `csSkyEnergy` to `atmosphereRadiance`.
- [ ] **Step 2:** `cargo test --release --locked sun_nee` → FAIL.
- [ ] **Step 3: Implement.** At each mesh PBR vertex with `SUN`: sample the cone, evaluate the PBR, and trace `skyOccluded`.
  - The NEE weight is `misWeight(sunPdf, ev.a)` when a BSDF continuation will be traced, i.e. `bounce + 1 < numBounces` and not `SKIP_FINAL_SCATTER`-skipped. Otherwise the weight is 1.
  - After `samplePbr`, set `sunMisWeight = misWeight(sampled.pdf, sunPdf(dir))`.
  - Camera rays and sphere-material continuations: `sunMisWeight = 1`.
- [ ] **Step 4:** whole suite → PASS. Check the bounce count 1 case (Review Focus 4) with the same test at `numBounces = 1`.
- [ ] **Step 5: Measure.**
  - Cost: the median GPU time of `--fastest --samples 24` at 640x400 on Sponza and Bistro, with and without `--no-sun`.
  - Noise: 1-spp images versus 1024-spp references.
  - Note the results in `performance/sky/README.md`. Commit `Sample the sun directly on mesh surfaces`.

### Task 3: Launcher

**Files:**
- Create: `src/launcher.rs`:
  - `Settings`, `Preset`, `Denoiser`, `Textures`;
  - `fn args(&self, scenes: &[SceneEntry]) -> Vec<String>`;
  - `parse`, `serialize`, `load`, `save`;
  - `fn scenes(root: &Path) -> Vec<SceneEntry>`;
  - `fn models(root: &Path) -> Vec<PathBuf>`;
  - `pub fn run()`: winit + egui window with `GpuContext` and `gui::Gui`.
- Modify: `src/main.rs`:
  - no arguments or `--launcher` → `launcher::run()`;
  - `--size`/`--fullscreen` set the initial window;
  - `GpuContext`/`Gui` become `pub(crate)` where the launcher needs them.
- Modify: `src/gui.rs`: a "Back to launcher" button that spawns `current_exe() --launcher` and closes.
- Create: `models/td-half16.{wgsl,bin}` and `models/td2-half24.{wgsl,bin}`, copied from `training/runs/pc/*/denoiser.*` and committed (about 60 KB).

**Interfaces:**
- `SceneEntry { name: String, args: Vec<String> }`. Entries:
  - from `performance/scenes/scenes.json` (`--model M --camera C`), each only if its model file exists;
  - plus "Bistro (night)" = bistro-street-textured + `--night`;
  - plus "Sphere demo" (no args).
- `Settings`: `scene`, `preset`, `size (u32,u32)`, `fullscreen`, `render_scale`, `denoiser`, `bounces`, `hw_rt`, `auto_tune`, `textures`, `sun_elevation`, `sun_azimuth`. Picking a preset fills render scale, denoiser and bounces; the user can then edit them.
- Presets: all add `--fastest`.

  | Preset | Render scale | Bounces | Denoiser | Max samples |
  |---|---|---|---|---|
  | Low | 0.5 | 2 | td-half16 | 1 |
  | Medium | 0.67 | 3 | td2-half24 | 1 |
  | High | 1.0 | 3 | td2-half24 | 1 |
  | Reference | 1.0 | 8 | no temporal | 4096 |

  A temporal denoiser is passed as `--temporal-model models/X.wgsl`.

- Other args:
  - textures: Auto adds nothing; Full adds `--ray-cones`; Off adds `--no-ray-cones`;
  - `--hw-rt` only when the adapter supports ray queries;
  - `--no-auto-tune`;
  - the sun: `--sun-elevation E --sun-azimuth A`, two new flags read in `scene_sky`.

- [ ] **Step 1: Write failing unit tests** in `src/launcher.rs`:
  - `presets_map_to_flags`: High → contains `--fastest`, `--temporal-model`, `models/td2-half24.wgsl`, and no `--render-scale`. Low → `--render-scale 0.5`.
  - `settings_roundtrip`: `parse(serialize(s)) == s`.
  - `settings_parse_ignores_garbage`: `parse("scene=\x00\nbounces=abc\nfoo=1")` returns defaults except valid lines.
  - `missing_models_are_hidden`: `scenes(tempdir)` lists only "Sphere demo"; `models(tempdir)` is empty, and the denoiser choices then offer only Off/Spatial.
- [ ] **Step 2:** `cargo test --release --locked launcher` → FAIL.
- [ ] **Step 3: Implement** the pure parts, then the egui window. The window has:
  - a scene list on the left, settings on the right, and a large Launch button;
  - `--hw-rt` shown only if supported;
  - an FPS hint line ("Low ≈ fastest, Reference = progressive, still camera").

  Launch saves the settings, spawns `current_exe()` with `args`, and exits.
- [ ] **Step 4:** whole suite → PASS. `cargo run --release` with no arguments opens the launcher; launch each preset on Bistro (night) and Sponza.
- [ ] **Step 5:** `cargo fmt`, zero warnings, update README (the launcher section). Commit `Add a launcher menu with quality presets and graphics settings`.

### Task 4: Deploy on the PC

- [ ] **Step 1:** push the branch.
- [ ] **Step 2:** On `rtx`:
  - `git clone -b fast-tracing-and-temporal-denoiser … ~/wgpu-raytracer-next`, or `git pull` if it exists;
  - link the old checkout's `assets/sponza`, `assets/scenes` and `training-venv` the way `pc_run.sh --from` does;
  - `cargo build --release --locked`.
- [ ] **Step 3:** headless check on the PC: render Bistro (night) and the Sponza sun (`--render-output`, `--hw-rt`) and copy the PNGs back. Expected: no panic, sun shadows visible.
- [ ] **Step 4:** report the command for the PC desktop: `cd ~/wgpu-raytracer-next && target/release/weekend-raytracer-wgpu`.
