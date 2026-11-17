#!/bin/bash
# Overnight run v3 (the RTX PC): training data from the current renderer and the joint
# denoise + upscale network at five quality tiers, finished by a wall-clock time.
#
#   cd ~/wgpu-raytracer-next && scripts/overnight_v3.sh [--end 07:00] [--smoke] [--detach]
#
# 1. build
# 2. data (training/runs/v3-*): every benchmark view plus night Bistro, with the sun,
#    6 moving triangle objects and 15% fast-motion sequences (25% in the evaluation set);
#    1-spp inputs as the real-time renderer makes them (2 bounces, ReSTIR lamps + GI) at
#    render scales 0.5 / 0.67 / 0.77, references with 4 bounces; per-scene calibration
#    picks RT cores or compute (--hw-rt). 60 fps video inputs (night Bistro, Sponza).
# 3. train tiny ... ultra in parallel, bf16 mixed precision (tensor cores), until 45 min
#    before --end
# 4. evaluate every tier per render scale, render videos, write v3-summary.txt
# Stages resume: finished captures are skipped. Log: overnight-v3.log (with --detach).
set -u
cd "$(dirname "$0")/.."
END=07:00 SMOKE=0 DETACH=0
while [ $# -gt 0 ]; do
    case $1 in
        --end) END=$2; shift 2 ;;
        --smoke) SMOKE=1; shift ;;
        --detach) DETACH=1; shift ;;
        *) echo "unknown argument $1"; exit 2 ;;
    esac
done
if [ $DETACH = 1 ]; then
    args=(--end "$END"); [ $SMOKE = 1 ] && args+=(--smoke)
    setsid nohup "$0" "${args[@]}" > overnight-v3.log 2>&1 < /dev/null &
    echo "running in the background (pid $!), log: $PWD/overnight-v3.log"
    exit 0
fi
stamp() { echo "=== $(date +%H:%M:%S) $*"; }
end_epoch=$(date -d "today $END" +%s)
[ "$end_epoch" -le "$(date +%s)" ] && end_epoch=$(date -d "tomorrow $END" +%s)
left_hours() {  # hours until --end minus a reserve (hours)
    python3 -c "print(max(0.05, ($end_epoch - $(date +%s)) / 3600 - $1))"
}
R=training/runs
mkdir -p $R/videos
PY=${PY:-$PWD/training-venv/bin/python}
[ -x "$PY" ] || PY=$PWD/training/.venv/bin/python
BIN=$PWD/target/release/weekend-raytracer-wgpu
stamp "start: until $(date -d @$end_epoch '+%a %H:%M'), python $PY"

# --- 1. build ---
cargo build --release --locked 2>&1 | tail -1 || exit 1

# --- 2. data ---
HWRT=""
nvidia-smi > /dev/null 2>&1 && HWRT="--hw-rt"
FLAGS="--restir --bounces 2 --reference-bounces 4"
SCALES=0.5,0.67,0.77
TRAIN=$R/v3-train EVAL=$R/v3-eval VIDEO=$R/v3-video
if [ $SMOKE = 1 ]; then
    TRAIN=$R/v3-smoke-train EVAL=$R/v3-smoke-eval VIDEO=$R/v3-smoke-video
    SCENES="--scenes fireplace-room,bistro-night"
    SEQ=4 FRAMES=3 SPP=16 SIZE=640x400 CROP=128 EVAL_SEQ=1 EVAL_FRAMES=3 EVAL_SPP=16 EVAL_SIZE=320x200 VIDEO_FRAMES=12
else
    SCENES=""
    SEQ=12 FRAMES=16 SPP=1024 SIZE=1280x720 CROP=320 EVAL_SEQ=2 EVAL_FRAMES=8 EVAL_SPP=1024 EVAL_SIZE=960x540 VIDEO_FRAMES=300
fi
capture() {  # out, capture.py arguments... (RT cores allowed; compute-only retry)
    local out=$1; shift
    python3 training/temporal/capture.py "$out" $SCENES "$@" --flags "$FLAGS $HWRT" || {
        [ -n "$HWRT" ] && stamp "capture with --hw-rt failed; compute only" &&
            python3 training/temporal/capture.py "$out" $SCENES "$@" --flags "$FLAGS"
    }
}
stamp "capture training data into $TRAIN"
capture $TRAIN --sequences $SEQ --frames $FRAMES --spp $SPP --size $SIZE --crop $CROP --seed-base 5000 \
    --motion-range 0.03,1 --render-scale $SCALES --objects 6 --fast-fraction 0.15 2>&1 | grep -vE "^glTF: optional"
stamp "capture evaluation set into $EVAL"
capture $EVAL --sequences $EVAL_SEQ --frames $EVAL_FRAMES --spp $EVAL_SPP --size $EVAL_SIZE --seed-base 9500 \
    --motion-range 0.05,0.12 --render-scale $SCALES --objects 6 --fast-fraction 0.25 2>&1 | grep -vE "^glTF: optional"
stamp "video inputs"
for spec in "bistro-night assets/scenes/exterior.glb -10,2,0,1,0,0.3,65 1.2 7101 --night" \
            "sponza assets/sponza/sponza-png.glb -8,1.6,0,1,0.08,-0.08,65 1.0 7201"; do
    set -- $spec
    [ -f $VIDEO/$1/meta.json ] && continue
    $BIN --model $2 --camera $3 --capture $VIDEO/$1 --scale $4 --sequences 1 --frames $VIDEO_FRAMES --no-reference \
        --size $EVAL_SIZE --seed $5 --motion-scale 0.08 --upscale-capture 0.5,0.67 --objects 6 --fastest \
        $FLAGS $HWRT ${6:-} 2>&1 | tail -1
done

# --- 3. training ---
HRS=$(left_hours 0.75)
STEPS=0
[ $SMOKE = 1 ] && HRS=0.05 && STEPS=20
stamp "train five tiers for $HRS h (bf16, tensor cores)"
train() {  # name, batch, args...: halves the batch and resumes on failure (out of memory)
    local name=$1 batch=$2; shift 2
    while [ "$batch" -ge 1 ]; do
        $PY training/upscale/train.py --data $TRAIN --out $R/$name --hours $HRS --batch $batch --crop 160 --amp \
            --eval-every $([ $SMOKE = 1 ] && echo 5 || echo 300) --max-steps $STEPS "$@" >> $R/$name.log 2>&1 && return 0
        echo "train $name failed at batch $batch" >> $R/$name.log
        batch=$((batch / 2))
    done
}
TIERS="tiny small medium large ultra"
P=v3
[ $SMOKE = 1 ] && P=v3-smoke
for tier in $TIERS; do
    train $P-$tier 6 --tier $tier &
done
wait

# --- 4. evaluation, videos, summary ---
for tier in $TIERS; do
    m=$R/$P-$tier
    [ -f $m/best.pt ] || continue
    stamp "evaluate $P-$tier"
    $PY training/upscale/evaluate.py --data $EVAL --weights $m/best.pt --out $m/eval --all > $m/eval.txt 2>&1
    for v in bistro-night sponza; do
        [ -d $VIDEO/$v ] || continue
        $PY training/upscale/upscale_video.py $VIDEO/$v 0 $m/best.pt $R/videos/$P-$v-$tier.mp4 --panel split 2>&1 | tail -1
    done
done
{
    echo "Overnight v3 $(date)"
    echo "Data: $TRAIN (inputs: 2 bounces + ReSTIR DI/GI, render scales $SCALES; references 4 bounces, $SPP spp)"
    for tier in $TIERS; do
        m=$R/$P-$tier
        [ -f $m/eval.txt ] || continue
        echo; echo "## $tier ($(grep -c '^step' $m/train.log 2>/dev/null) evaluations, $(python3 -c "import json; print(json.load(open('$m/summary.json'))['steps'])" 2>/dev/null) steps)"
        sed -n '/mean over scenes/,$p' $m/eval.txt
    done
    echo; echo "Videos: $R/videos/$P-*.mp4 (left: raw 1-spp samples, right: network)"
} > $R/$P-summary.txt
cat $R/$P-summary.txt
stamp "done"
