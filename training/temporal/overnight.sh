#!/bin/bash
# Unattended run on the RTX PC: slow-motion (60 fps-like) training data, retraining,
# evaluation, shader export and videos. Everything ends before $END.
#   nohup setsid training/temporal/overnight.sh HOURS > training/runs/overnight.log 2>&1 &
cd "$(dirname "$0")/../.."
PY=$PWD/training-venv/bin/python
R=training/runs
END=$(python3 -c "import time; print(int(time.time() + ${1:-7.6} * 3600))")
stamp() { echo "=== $(date +%H:%M:%S) $*"; }
stamp "start, end at $(date -d @$END +%H:%M)"

# 1. Training data: 16-frame clips, motion per sequence log-uniform 0.03..1 of the
#    original training motion (0.08 is the 60 fps video), 384 crops of 720p, 2048 spp.
stamp "capture training data"
timeout 3h python3 training/temporal/capture.py $R/temporal-slow --sequences 24 --frames 16 --spp 2048 \
    --size 1280x720 --crop 384 --seed-base 3000 --motion-range 0.03,1 --extra=--hw-rt
# Evaluation: full 720p frames, 60 fps-like motion, unseen seeds.
stamp "capture slow evaluation set"
timeout 75m python3 training/temporal/capture.py $R/temporal-eval-slow --sequences 2 --frames 16 --spp 2048 \
    --size 1280x720 --seed-base 9000 --motion-range 0.05,0.12 --extra=--hw-rt
# Video inputs (no reference): 5 s at 60 fps.
stamp "capture video inputs"
for spec in "sponza-atrium assets/sponza/sponza-png.glb -8,1.6,0,1,0.08,-0.08,65 1.0 7201" \
            "living-room assets/scenes/living_room.glb 0.3,1.4,7.5,0,-0.1,-1,65 0.4 7301"; do
    set -- $spec
    [ -f $R/video-$1/meta.json ] || timeout 10m target/release/weekend-raytracer-wgpu --model $2 --camera $3 \
        --capture $R/video-$1 --scale $4 --sequences 1 --frames 300 --no-reference --size 1280x720 \
        --seed $5 --motion-scale 0.08 --fastest --hw-rt 2>&1 | tail -1
done

# 2. Training until 70 minutes before the end: fine-tune the current model, and a wider one.
HOURS=$(python3 -c "import time; print(max(0.5, ($END - time.time() - 70*60) / 3600))")
stamp "train for $HOURS h"
train() {  # name, batch, args...; halves the batch and resumes on failure (e.g. out of memory)
    local name=$1 batch=$2; shift 2
    while [ "$batch" -ge 2 ]; do
        (cd training/temporal && $PY train.py --data ../runs/temporal-slow --out ../runs/$name --hours $HOURS \
            --batch $batch --crop 192 --eval-every 300 --threads 4 "$@") >> $R/$name.log 2>&1 && return 0
        echo "train $name failed at batch $batch" >> $R/$name.log
        batch=$((batch / 2))
    done
}
train td2-half16 8 --channels 16 --half --init ../runs/td-half16/best.pt --lr 1e-3 &
train td2-half24 8 --channels 24 --half &
wait

# 3. Evaluation, export, videos.
for m in td-half16 td2-half16 td2-half24; do
    [ -f $R/$m/best.pt ] || continue
    stamp "evaluate $m"
    $PY training/temporal/evaluate.py --data $R/temporal-eval --weights $R/$m/best.pt --out $R/$m/eval-full --all > $R/$m/eval-full.txt 2>&1
    $PY training/temporal/evaluate.py --data $R/temporal-eval-slow --weights $R/$m/best.pt --out $R/$m/eval-slow --all > $R/$m/eval-slow.txt 2>&1
    $PY training/temporal/export.py $R/$m/best.pt $R/$m/denoiser.wgsl > /dev/null 2>&1
done
mkdir -p $R/videos
for m in td-half16 td2-half16 td2-half24; do
    [ -f $R/$m/best.pt ] || continue
    for v in "bistro video-bistro/seq_001" "sponza video-sponza-atrium/seq_000" "living video-living-room/seq_000"; do
        set -- $v
        [ -d $R/$2 ] || continue
        [ "$(date +%s)" -lt "$END" ] || break
        stamp "video $m $1"
        $PY training/temporal/denoise_video.py $R/$2 $R/$m/best.pt $R/videos/$1-$m.mp4 | tail -1
    done
done
[ -f $R/td2-half16/best.pt ] && $PY training/temporal/denoise_video.py $R/video-bistro/seq_001 $R/td2-half16/best.pt \
    $R/videos/bistro-td2-half16-split.mp4 --panel split | tail -1

# 4. Summary.
$PY - <<'PY' > $R/overnight-summary.txt
import json, pathlib
R = pathlib.Path("training/runs")
for m in ("td-half16", "td2-half16", "td2-half24"):
    for s in ("eval-full", "eval-slow"):
        f = R / m / s / "metrics.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        mean = lambda k, meth: sum(v[meth][k] for v in d.values()) / len(d)
        print(f"{m:11} {s:9} PSNR raw {mean('psnr','raw'):.2f} ema {mean('psnr','ema'):.2f} net {mean('psnr','network'):.2f}"
              f" | flicker raw {mean('flicker','raw'):.4f} net {mean('flicker','network'):.4f}")
for m in ("td2-half16", "td2-half24"):
    f = R / m / "summary.json"
    if f.exists():
        print(m, f.read_text().replace("\n", ""))
PY
cat $R/overnight-summary.txt
stamp "done"
