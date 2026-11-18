#!/bin/bash
# Overnight run v4 (the RTX PC): training data from the current renderer with finished
# references, then the temporal denoisers retrained and the fast network trained.
#
#   cd ~/wgpu-raytracer-next && scripts/overnight_v4.sh [--end 07:00] [--smoke] [--detach]
#
# 1. build
# 2. data (training/runs/v4-*): every benchmark view plus night Bistro, 6 moving
#    objects, 15% fast-motion sequences; 1-spp inputs as the real-time renderer traces
#    them (2 bounces, ReSTIR lamps + bounce reuse) at full size and at render scales
#    0.5 / 0.67, as 320-pixel windows of 720p, 1080p, 1440p and 4K frames;
#    references: the plain path tracer, 2 bounces, 1024 spp in 8 groups,
#    finished by training/reference/finish_capture.py. Compute only (RT-core captures
#    hung the GPU twice on 2026-11-18).
# 3. train until 30 min before --end, in parallel: td-half16, td2-half24
#    (training/temporal) and, when training/fast/train.py exists, the fast network.
# 4. export the denoisers (f32 and f16), evaluate, write v4-summary.txt
# Stages resume: finished captures are skipped. Log: overnight-v4.log (with --detach).
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
    setsid nohup "$0" "${args[@]}" > overnight-v4.log 2>&1 < /dev/null &
    echo "running in the background (pid $!), log: $PWD/overnight-v4.log"
    exit 0
fi
stamp() { echo "=== $(date +%H:%M:%S) $*"; }
end_epoch=$(date -d "today $END" +%s)
[ "$end_epoch" -le "$(date +%s)" ] && end_epoch=$(date -d "tomorrow $END" +%s)
left_hours() { python3 -c "print(max(0.05, ($end_epoch - $(date +%s)) / 3600 - $1))"; }
R=training/runs
PY=${PY:-$PWD/training-venv/bin/python}
[ -x "$PY" ] || PY=$PWD/training/.venv/bin/python
BIN=$PWD/target/release/weekend-raytracer-wgpu
stamp "start: until $(date -d @$end_epoch '+%a %H:%M'), python $PY"
cargo build --release --locked 2>&1 | tail -1 || exit 1

FLAGS="--restir --bounces 2 --reference-bounces 2 --reference-groups 8"
SCALES=0.5,0.67
# Output resolutions: windows of 720p, 1080p, 1440p and 4K frames (a window costs the
# same at any frame size), so the networks see pixels of every size on screen.
if [ $SMOKE = 1 ]; then
    P=v4-smoke SCENES="--scenes fireplace-room,sponza-atrium"
    SIZES="640x400 1280x720" SEQ=3 FRAMES=3 SPP=64 CROP=128 EVAL_SEQ=1 EVAL_FRAMES=3 EVAL_SIZE=320x200
else
    P=v4 SCENES=""
    SIZES="1280x720 1920x1080 2560x1440 3840x2160" SEQ=${SEQ:-3} FRAMES=16 SPP=1024 CROP=320 EVAL_SEQ=2 EVAL_FRAMES=8 EVAL_SIZE=960x540
fi
EVAL=$R/$P-eval
TRAIN=""
for size in $SIZES; do TRAIN="$TRAIN${TRAIN:+,}$R/$P-train-${size#*x}"; done
gpu_ok() { nvidia-smi > /dev/null 2>&1 && $PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; }

seed=11000
for size in $SIZES; do
    out=$R/$P-train-${size#*x}
    stamp "capture training data into $out"
    python3 training/temporal/capture.py $out $SCENES --sequences $SEQ --frames $FRAMES --spp $SPP --size $size --crop $CROP \
        --seed-base $seed --motion-range 0.03,1 --render-scale $SCALES --objects 6 --fast-fraction 0.15 --flags "$FLAGS" 2>&1 | grep -vE "^glTF: optional|^SBVH|^Textures|^glTF"
    gpu_ok || { stamp "GPU fault during the training capture"; exit 1; }
    $PY training/reference/finish_capture.py $out
    seed=$((seed + 100))
done
stamp "capture evaluation set into $EVAL"
python3 training/temporal/capture.py $EVAL $SCENES --sequences $EVAL_SEQ --frames $EVAL_FRAMES --spp $SPP --size $EVAL_SIZE \
    --seed-base 12500 --motion-range 0.05,0.12 --render-scale $SCALES --objects 6 --fast-fraction 0.25 --flags "$FLAGS" 2>&1 | grep -vE "^glTF: optional|^SBVH|^Textures|^glTF"
gpu_ok || { stamp "GPU fault after the evaluation capture"; exit 1; }
$PY training/reference/finish_capture.py $EVAL

HRS=$(left_hours 0.5)
STEPS=""
[ $SMOKE = 1 ] && HRS=0.03
stamp "train for $HRS h: td-half16, td2-half24$([ -f training/fast/train.py ] && echo ', fast network')"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TD_DATA=$(echo "$TRAIN" | sed 's|training/runs|../../training/runs|g')
(cd training/temporal && $PY train.py --data $TD_DATA --out ../../$R/$P-td-half16 --hours $HRS --channels 16 --half --val-sequences 1 > ../../$R/$P-td-half16.log 2>&1) &
(cd training/temporal && $PY train.py --data $TD_DATA --out ../../$R/$P-td2-half24 --hours $HRS --channels 24 --half --val-sequences 1 > ../../$R/$P-td2-half24.log 2>&1) &
if [ -f training/fast/train.py ]; then
    $PY training/fast/train.py --data $TRAIN --out $R/$P-fast --hours $HRS > $R/$P-fast.log 2>&1 &
fi
wait

stamp "export and evaluate"
{
    echo "v4 run, finished $(date '+%F %H:%M')"
    for run in td-half16 td2-half24; do
        m=$R/$P-$run
        [ -f $m/best.pt ] || { echo "$run: no model"; continue; }
        $PY training/temporal/export.py $m/best.pt $m/denoiser.wgsl --f16 > /dev/null 2>&1
        echo "$run: $(tail -1 $R/$P-$run.log)"
    done
    [ -f $R/$P-fast.log ] && echo "fast: $(tail -1 $R/$P-fast.log)"
} > $R/$P-summary.txt 2>&1
cat $R/$P-summary.txt
stamp "done"
