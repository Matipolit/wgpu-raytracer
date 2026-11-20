#!/bin/bash
# Install this checkout's release build on the Steam Deck (ssh alias `deck`,
# ~/wgpu-raytracer-test; the Deck has no Rust toolchain, so the laptop builds for it).
#
#   scripts/deck_install.sh [NAME=RUN_DIR ...]
#
# NAME=RUN_DIR also installs a temporal denoiser exported into RUN_DIR
# (denoiser.wgsl/.bin and the .f16 pair) as models/NAME.*, the names the launcher's
# presets use (td-half16: Low, td2-half24: Medium and High). The models it replaces are
# kept once in models-old/.
set -e
cd "$(dirname "$0")/.."
D=wgpu-raytracer-test
cargo build --release --locked 2>&1 | tail -1
scp -q target/release/weekend-raytracer-wgpu deck:$D/target/release/weekend-raytracer-wgpu.new
ssh deck "cd $D/target/release && chmod +x weekend-raytracer-wgpu.new && mv -f weekend-raytracer-wgpu.new weekend-raytracer-wgpu && rm -f weekend-raytracer-wgpu.bin"
for model in "$@"; do
    name=${model%%=*}; run=${model#*=}
    ssh deck "cd $D && mkdir -p models-old && { [ -f models-old/$name.wgsl ] || cp -a models/$name.* models-old/ 2>/dev/null || true; }"
    for e in wgsl bin f16.wgsl f16.bin; do scp -q $run/denoiser.$e deck:$D/models/$name.$e; done
    echo "installed $run as models/$name"
done
ssh deck "cd $D && ls -la target/release models"
