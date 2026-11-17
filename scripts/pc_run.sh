#!/bin/bash
# Unattended GPU run (the RTX PC): everything the laptop could not do, in one go.
#
#   git clone -b fast-tracing-and-temporal-denoiser https://github.com/Matipolit/wgpu-raytracer ~/wgpu-raytracer-next
#   cd ~/wgpu-raytracer-next && scripts/pc_run.sh --from ~/wgpu-raytracer --hours 8
#
# --from DIR  an older checkout whose assets/scenes, Sponza model, training/runs and
#             training-venv are reused (symlinked); omit when this checkout already has them.
# --hours H   total budget (default 8); training gets what the other stages leave.
# --detach    keep running after the terminal closes (log: pc-run.log in the repository root).
# --smoke     tiny everything (minutes), to check the script itself.
#
# Stages (each skipped when its output exists, so a rerun resumes):
#   1. build
#   2. re-export last night's denoisers with the current exporter + PyTorch parity check
#   3. RTX: compute vs hardware ray queries on every benchmark view
#   4. denoise + upscale data at render scales 0.5, 0.6, 0.67, 0.77: added to last night's
#      2048-spp captures (--low-only, minutes) or captured from scratch if those are missing;
#      evaluation set; 5 s 60 fps video inputs
#   5. train two joint denoise + upscale networks (16 and 24 channels) in parallel
#   6. evaluate per scale, render videos, write training/runs/pc-summary.txt
set -u
cd "$(dirname "$0")/.."
FROM="" HOURS=8 SMOKE=0
args=("$@")
while [ $# -gt 0 ]; do
    case $1 in
        --from) FROM=$2; shift 2 ;;
        --hours) HOURS=$2; shift 2 ;;
        --smoke) SMOKE=1; shift ;;
        --detach) shift ;;
        *) echo "unknown argument $1"; exit 2 ;;
    esac
done
if printf '%s\n' "${args[@]}" | grep -qx -- --detach; then
    rest=()
    for x in "${args[@]}"; do [ "$x" != --detach ] && rest+=("$x"); done
    nohup setsid "$0" "${rest[@]}" > pc-run.log 2>&1 < /dev/null &
    echo "running in the background; follow with: tail -f $PWD/pc-run.log"
    exit 0
fi

R=training/runs
END=$(python3 -c "import time; print(int(time.time() + $HOURS * 3600))")
stamp() { echo "=== $(date +%H:%M:%S) $*"; }
left_hours() { python3 -c "import time; print(max(0.05, ($END - time.time()) / 3600 - $1))"; }

# --- 0. data and environment from an older checkout ---
if [ -n "$FROM" ]; then
    for path in assets/scenes training/runs training-venv; do
        # Link when this checkout lacks it (or has only an empty directory).
        if [ -e "$FROM/$path" ] && { [ ! -e "$path" ] || [ -z "$(ls -A "$path" 2>/dev/null)" ]; }; then
            rm -rf "$path"
            ln -s "$(realpath "$FROM/$path")" "$path"
            echo "linked $path -> $FROM/$path"
        fi
    done
    for model in "$FROM"/assets/sponza/*.glb; do
        [ -e "$model" ] && [ ! -e "assets/sponza/$(basename "$model")" ] && ln -s "$(realpath "$model")" assets/sponza/
    done
fi
mkdir -p training/runs
PY=${PY:-$PWD/training-venv/bin/python}
[ -x "$PY" ] || PY=$PWD/training/.venv/bin/python
[ -x "$PY" ] || { echo "no Python environment with torch (training-venv or training/.venv); set PY=..."; exit 1; }
BIN=$PWD/target/release/weekend-raytracer-wgpu
SCALES=0.5,0.6,0.67,0.77
if [ $SMOKE = 1 ]; then
    SEQ=3 FRAMES=3 SPP=16 SIZE=640x400 CROP=128 EVAL_SIZE=640x400 VIDEO_FRAMES=12 SCENES=sponza-atrium,fireplace-room
else
    SEQ=24 FRAMES=16 SPP=2048 SIZE=1280x720 CROP=384 EVAL_SIZE=1280x720 VIDEO_FRAMES=300 SCENES=""
fi
scene_arg() { [ -n "$SCENES" ] && echo "--scenes $SCENES"; }
HWRT=""
nvidia-smi > /dev/null 2>&1 && HWRT="--extra=--hw-rt"
stamp "start: budget $HOURS h, python $PY, ${HWRT:-no hardware ray queries}"

# --- 1. build ---
stamp "build"
source "$HOME/.cargo/env" 2>/dev/null
cargo build --release --locked 2>&1 | tail -2 || exit 1

# --- 2. last night's temporal denoisers: export with the current exporter, parity ---
for run in td-half16 td2-half16 td2-half24; do
    [ -f $R/$run/best.pt ] || continue
    [ -f $R/$run/parity.txt ] && continue
    stamp "export + parity: $run"
    $PY training/temporal/export.py $R/$run/best.pt $R/$run/denoiser.wgsl > /dev/null
    rm -rf $R/$run/parity
    $BIN --model assets/sponza/sponza-png.glb --camera -8,1.6,0,1,0.08,-0.08,65 --capture $R/$run/parity --scale 1 \
        --sequences 1 --frames 6 --size 320x200 --fastest --no-reference --temporal-model $R/$run/denoiser.wgsl > /dev/null 2>&1
    $PY training/temporal/parity.py $R/$run/parity/seq_000 $R/$run/best.pt 320 200 | tail -1 | tee $R/$run/parity.txt
done

# --- 3. RTX: compute vs hardware ray queries ---
if [ -n "$HWRT" ] && [ ! -f performance/scenes/results/rtx-hwrt/results.json ] && [ $SMOKE = 0 ]; then
    stamp "benchmark compute vs --hw-rt"
    python3 performance/scenes/bench.py --out performance/scenes/results/rtx-hwrt --reps 2 \
        --config compute=$BIN:--fastest,--no-auto-tune --config hwrt=$BIN:--fastest,--no-auto-tune,--hw-rt 2>&1 | tail -12
fi

# --- 4. denoise + upscale data ---
# Training: last night's 2048-spp capture (same arguments and seeds) + low inputs.
TRAIN=$R/temporal-slow
if [ $SMOKE = 1 ]; then TRAIN=$R/upscale-smoke; fi
if [ ! -d $TRAIN ] || [ -z "$(ls $TRAIN/*/meta.json 2>/dev/null)" ]; then
    stamp "capture training data from scratch into $TRAIN"
    python3 training/temporal/capture.py $TRAIN $(scene_arg) --sequences $SEQ --frames $FRAMES --spp $SPP --size $SIZE \
        --crop $CROP --seed-base 3000 --motion-range 0.03,1 --render-scale $SCALES $HWRT
else
    stamp "add low-resolution inputs to $TRAIN"
    if ! python3 training/temporal/capture.py $TRAIN $(scene_arg) --sequences $SEQ --frames $FRAMES --spp $SPP --size $SIZE \
        --crop $CROP --seed-base 3000 --motion-range 0.03,1 --render-scale $SCALES --low-only $HWRT; then
        # The existing capture does not match (arguments or seeds): capture from scratch.
        TRAIN=$R/upscale-data
        stamp "low-only failed; capture training data from scratch into $TRAIN"
        python3 training/temporal/capture.py $TRAIN $(scene_arg) --sequences $SEQ --frames $FRAMES --spp $SPP --size $SIZE \
            --crop $CROP --seed-base 3000 --motion-range 0.03,1 --render-scale $SCALES $HWRT
    fi
fi
EVAL=$R/temporal-eval-slow
if [ $SMOKE = 1 ]; then EVAL=$R/upscale-smoke-eval; fi
if [ ! -d $EVAL ] || [ -z "$(ls $EVAL/*/meta.json 2>/dev/null)" ]; then
    stamp "capture evaluation set into $EVAL"
    python3 training/temporal/capture.py $EVAL $(scene_arg) --sequences 2 --frames $FRAMES --spp $SPP --size $EVAL_SIZE \
        --seed-base 9000 --motion-range 0.05,0.12 --render-scale $SCALES $HWRT
else
    stamp "add low-resolution inputs to $EVAL"
    python3 training/temporal/capture.py $EVAL $(scene_arg) --sequences 2 --frames $FRAMES --spp $SPP --size $EVAL_SIZE \
        --seed-base 9000 --motion-range 0.05,0.12 --render-scale $SCALES --low-only $HWRT
fi
stamp "video inputs"
for spec in "bistro assets/scenes/exterior.glb -10,2,0,1,0,0.3,65 1.2 7101" \
            "sponza assets/sponza/sponza-png.glb -8,1.6,0,1,0.08,-0.08,65 1.0 7201" \
            "living assets/scenes/living_room.glb 0.3,1.4,7.5,0,-0.1,-1,65 0.4 7301"; do
    set -- $spec
    [ -f $R/upscale-video/$1/meta.json ] && continue
    [ -e "$2" ] || continue
    $BIN --model $2 --camera $3 --capture $R/upscale-video/$1 --scale $4 --sequences 1 --frames $VIDEO_FRAMES --no-reference \
        --size $EVAL_SIZE --seed $5 --motion-scale 0.08 --upscale-capture 0.5,0.67 --fastest ${HWRT#--extra=} 2>&1 | tail -1
done

# --- 5. training ---
HRS=$(left_hours 0.75)
STEPS=0
[ $SMOKE = 1 ] && HRS=0.05 && STEPS=20
stamp "train for $HRS h"
train() {  # name, batch, args...: halves the batch and resumes on failure (out of memory)
    local name=$1 batch=$2; shift 2
    while [ "$batch" -ge 1 ]; do
        $PY training/upscale/train.py --data $TRAIN --out $R/$name --hours $HRS --batch $batch --crop 160 \
            --eval-every $([ $SMOKE = 1 ] && echo 5 || echo 300) --max-steps $STEPS "$@" >> $R/$name.log 2>&1 && return 0
        echo "train $name failed at batch $batch" >> $R/$name.log
        batch=$((batch / 2))
    done
}
train up-half16 6 --channels 16 &
train up-half24 6 --channels 24 &
wait

# --- 6. evaluation, videos, summary ---
for m in up-half16 up-half24; do
    [ -f $R/$m/best.pt ] || continue
    stamp "evaluate $m"
    $PY training/upscale/evaluate.py --data $EVAL --weights $R/$m/best.pt --out $R/$m/eval --all > $R/$m/eval.txt 2>&1
    for v in bistro sponza living; do
        [ -d $R/upscale-video/$v ] || continue
        $PY training/upscale/upscale_video.py $R/upscale-video/$v 0 $R/$m/best.pt $R/videos/upscale-$v-$m.mp4 --panel split 2>&1 | tail -1
    done
done
{
    echo "PC run $(date)"
    echo; echo "## Temporal denoisers re-exported (PyTorch parity)"
    for run in td-half16 td2-half16 td2-half24; do [ -f $R/$run/parity.txt ] && echo "$run: $(cat $R/$run/parity.txt)"; done
    [ -f $R/overnight-summary.txt ] && { echo; echo "## Last night"; cat $R/overnight-summary.txt; }
    echo; echo "## RTX compute vs --hw-rt"
    [ -f performance/scenes/results/rtx-hwrt/results.json ] && python3 -c "
import json; d = json.load(open('performance/scenes/results/rtx-hwrt/results.json'))
for r in d['rows']:
    if r['config'] == 'hwrt': print(f\"{r['scene']:24} {r['speedup']:.2f}x  ({r['ms']:.1f} ms)\")
print('geomean', round(d['summary']['hwrt']['geomean_speedup'], 3))"
    for m in up-half16 up-half24; do
        [ -f $R/$m/eval.txt ] && { echo; echo "## Denoise + upscale $m (evaluation set)"; sed -n '/mean over scenes/,$p' $R/$m/eval.txt; }
    done
    echo; echo "Videos: $R/videos/upscale-*.mp4 (left: raw 1-spp samples, right: network)"
} > $R/pc-summary.txt
cat $R/pc-summary.txt
stamp "done"
