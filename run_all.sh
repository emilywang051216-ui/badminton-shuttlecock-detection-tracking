#!/usr/bin/env bash
# One-command full pipeline for a Linux GPU box. Leave it running.
#
#   ./run_all.sh            # FULL: 4 YOLO folds + 3 TrackNet folds + eval + demos + benchmark
#   ./run_all.sh core       # CORE: YOLO {random,lovo_test3} + TrackNet fold3 + eval + demo + bench
#   FORCE=1 ./run_all.sh    # retrain even if a model's best.pt already exists
#
# Tunables (env vars): BATCH, TN_BATCH, CACHE, EPOCHS_YOLO, EPOCHS_TN, DEVICE
# Resumable: already-trained models are skipped unless FORCE=1, so a re-run continues where it stopped.
set -uo pipefail
cd "$(dirname "$0")"                       # project root
mkdir -p outputs/logs

SCOPE="${1:-full}"
BATCH="${BATCH:-32}"                        # YOLO batch; lower it if you run into VRAM limits
TN_BATCH="${TN_BATCH:-16}"                  # TrackNet batch
CACHE="${CACHE:-ram}"                       # RAM cache the frames on a big-RAM host (auto-disables if short)
EPOCHS_YOLO="${EPOCHS_YOLO:-100}"           # early stopping (patience) usually ends well before this
EPOCHS_TN="${EPOCHS_TN:-30}"
DEVICE="${DEVICE:-0}"

if [ "$SCOPE" = "core" ]; then
  YOLO_SPLITS=(random lovo_test3)
  TN_FOLDS=(3)
elif [ "$SCOPE" = "deploy" ]; then
  # Deployment models trained on ALL 3 videos (for running on a genuinely new clip).
  YOLO_SPLITS=(all)
  TN_FOLDS=(all)
else
  YOLO_SPLITS=(random lovo_test1 lovo_test2 lovo_test3)
  TN_FOLDS=(1 2 3)
fi

echo "=================================================================="
echo " Shuttlecock full run  |  scope=$SCOPE  batch=$BATCH cache=$CACHE"
echo "   YOLO splits : ${YOLO_SPLITS[*]}   (epochs=$EPOCHS_YOLO)"
echo "   TrackNet    : folds ${TN_FOLDS[*]}   (epochs=$EPOCHS_TN, batch=$TN_BATCH)"
echo "   device=$DEVICE  FORCE=${FORCE:-0}"
echo "=================================================================="

run() {                                     # run <logname> <command...>
  local name="$1"; shift
  echo ">>> [$(date +%H:%M:%S)] $name ..."
  if "$@" > "outputs/logs/$name.log" 2>&1; then
    echo "    OK   -> outputs/logs/$name.log"
  else
    echo "    FAILED -> outputs/logs/$name.log (last 25 lines):"
    tail -25 "outputs/logs/$name.log"
    exit 1
  fi
}

SECONDS=0

# 0 — data prep (idempotent: extracts + regenerates split yamls/txt/csv with LOCAL paths)
run 01_prepare python scripts/01_prepare_data.py

# 1 — YOLOv11 detectors
for split in "${YOLO_SPLITS[@]}"; do
  if [ -z "${FORCE:-}" ] && [ -f "models/yolo_${split}/weights/best.pt" ]; then
    echo ">>> yolo_${split} already trained — skipping (FORCE=1 to redo)"; continue
  fi
  run "yolo_${split}" python scripts/02_train_yolo.py --split "$split" \
      --batch "$BATCH" --cache "$CACHE" --epochs "$EPOCHS_YOLO" --device "$DEVICE"
done

# 2 — TrackNetV2 (workers>0, no --cache: each worker would copy the cache)
for v in "${TN_FOLDS[@]}"; do
  if [ "$v" = "all" ]; then tndir="models/tracknet_all"; else tndir="models/tracknet_test${v}"; fi
  if [ -z "${FORCE:-}" ] && [ -f "${tndir}/best.pt" ]; then
    echo ">>> ${tndir} already trained — skipping (FORCE=1 to redo)"; continue
  fi
  run "tracknet_${v}" python scripts/03_train_tracknet.py --test-video "$v" \
      --batch "$TN_BATCH" --epochs "$EPOCHS_TN" --device cuda
done

if [ "$SCOPE" = "deploy" ]; then
  echo "=================================================================="
  echo " DEPLOY MODELS DONE in $((SECONDS/60))m$((SECONDS%60))s"
  echo "   models/yolo_all/weights/best.pt   models/tracknet_all/best.pt"
  echo "   (trained on ALL 3 videos — for running on a genuinely new clip)"
  echo "   Run on a new video:"
  echo "     python scripts/05_infer_video.py --model tracknet --source NEW.mp4 \\"
  echo "            --weights models/tracknet_all/best.pt"
  echo "=================================================================="
  exit 0
fi

# 3 — evaluate + comparison table + gap/curve charts
run 04_eval python scripts/04_eval_compare.py

# 4 — demos on the held-out video 3 (never seen in lovo_test3 training)
run 05_demo_yolo     python scripts/05_infer_video.py --model yolo     --source dataset:3
if [ -f "models/tracknet_test3/best.pt" ]; then
  run 05_demo_tracknet python scripts/05_infer_video.py --model tracknet --source dataset:3
fi

# 5 — ONNX export + FPS benchmark
run 06_bench python scripts/06_export_benchmark.py --weights models/yolo_lovo_test3/weights/best.pt

echo "=================================================================="
echo " ALL DONE in $((SECONDS/60))m$((SECONDS%60))s"
echo "   charts  -> outputs/gap_chart.png, yolo_curves.png, tracknet_curves.png"
echo "   tables  -> outputs/eval_yolo.csv, eval_tracknet.csv, benchmark.csv"
echo "   demos   -> outputs/demo_*.mp4, heatmap_*.png"
echo "   weights -> models/*/"
echo "=================================================================="
