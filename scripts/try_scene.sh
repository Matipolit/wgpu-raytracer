#!/bin/sh
# Open a benchmark scene interactively with a chosen renderer configuration.
#
#   scripts/try_scene.sh SCENE [CONFIG] [extra renderer flags...]
#
# SCENE:  a name from performance/scenes/scenes.json (run without arguments to list them).
# CONFIG: fastest (default)  --fastest: SBVH, 4-wide BVH, shared-memory stack, leaf size 1, plus previous
#         previous           --sah-stop --skip-final-scatter (the earlier best)
#         default            no optimization flags
# Extra flags pass through, e.g. --bounces 8, --size, --no-auto-exposure.
set -eu
cd "$(dirname "$0")/.."
SCENES=performance/scenes/scenes.json
if [ $# -lt 1 ]; then
    echo "usage: $0 SCENE [fastest|previous|default] [flags...]"
    echo "scenes:"
    python3 -c "import json; [print('  ' + k) for k in json.load(open('$SCENES'))]"
    exit 2
fi
scene=$1
shift
config=fastest
case "${1:-}" in fastest|previous|default) config=$1; shift ;; esac
model=$(python3 -c "import json,sys; print(json.load(open('$SCENES'))[sys.argv[1]]['model'])" "$scene")
camera=$(python3 -c "import json,sys; print(json.load(open('$SCENES'))[sys.argv[1]]['camera'])" "$scene")
if [ ! -f "$model" ]; then
    echo "$model is missing; see performance/scenes/README.md for downloads." >&2
    exit 1
fi
cargo build --release --locked
case $config in
    fastest) set -- --fastest "$@" ;;
    previous) set -- --sah-stop --skip-final-scatter "$@" ;;
    default) ;;
esac
echo "$scene ($config): $model"
exec target/release/weekend-raytracer-wgpu --model "$model" --camera "$camera" "$@"
